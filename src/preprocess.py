"""
preprocess.py (shared feature engineering for Cat Profile Optimiser).

The SAME preprocess() path is used for training (a DataFrame of many cats)
and inference (one cat from the Streamlit form). This guarantees no
train/inference skew.

Categorical handling: Breed1 and Color1 use XGBoost's native categorical
support. The set of valid categories comes from the PetFinder label CSVs so a breed seen in the form but
not in the training split won't break inference.
"""


from pathlib import Path
import pandas as pd
import json


# === Paths ===
# Resolve relative to this file so it works from notebooks, app, or scripts.

_HERE = Path(__file__).resolve().parent
_DATA_RAW = _HERE.parent / "data" / "raw"
BREED_LABELS_PATH = _DATA_RAW / "BreedLabels.csv"
COLOR_LABELS_PATH = _DATA_RAW / "ColorLabels.csv"


# === Features ===
# Order matters: the model is trained on columns in exactly this order, and inference must reproduce it.

CATEGORICAL_FEATURES = ["Breed1", "Color1"]

NUMERIC_FEATURES = [
    "Age", "MaturitySize", "FurLength",
    "Vaccinated", "Dewormed", "Sterilized", "Health",
    "Fee", "PhotoAmt",
]

DERIVED_FEATURES = ["desc_word_count", "is_free"]

FEATURE_ORDER = CATEGORICAL_FEATURES + NUMERIC_FEATURES + DERIVED_FEATURES

TARGET = "adopted_fast"


# === Category Sets ===
def load_category_sets() -> dict[str, list[int]]:
    """Return the complete set of valid category IDs for Breed1 and Color1,
    sourced from the PetFinder label files.

    Breeds are filtered to cats (Type == 2). A 0 is prepended to each set to
    represent 'not specified / unknown', which appears in the data and may
    come from the form.

    Returns e.g. {"Breed1": [0, 241, 242, ...], "Color1": [0, 1, 2, ...]}.
    The model and inference both apply these exact sets so the categorical
    encoding never drifts.
    """

    df_breeds = pd.read_csv(BREED_LABELS_PATH)
    df_colors = pd.read_csv(COLOR_LABELS_PATH)

    cat_breed_ids = df_breeds.loc[df_breeds["Type"] == 2, "BreedID"].tolist()
    color_ids = df_colors["ColorID"].tolist()

    return {
        "Breed1": [0] + sorted(cat_breed_ids),
        "Color1": [0] + sorted(color_ids),
    }


# === Derived Features ===
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add desc_word_count and is_free. Operates on a copy.

    Mirrors the EDA derivations exactly so training and inference agree:
      - desc_word_count: number of whitespace-separated words in Description
                         (missing/empty description -> 0)
      - is_free:         1 if Fee == 0 else 0
    """

    df = df.copy()

    desc = df["Description"].fillna("") if "Description" in df.columns else ""
    df["desc_word_count"] = pd.Series(desc, index=df.index).astype(str).str.split().apply(len)

    df["is_free"] = (df["Fee"] == 0).astype(int)

    return df


# === Main Preprocess Function ===
def preprocess(raw, category_sets: dict | None = None) -> pd.DataFrame:
    """Transform raw cat data into model-ready features.

    THE SAME PATH for training and inference (prevents skew).

    === Parameters ===
    raw : pd.DataFrame | dict
        Either many cats (training) or one cat from the form (a dict).
    category_sets : dict, optional
        Valid category IDs for Breed1/Color1. Loaded from label CSVs if
        not provided. Pass it in to avoid re-reading the CSVs every call
        (e.g. at inference, load once at app startup).

    === Returns ===
    pd.DataFrame
        Columns exactly == FEATURE_ORDER, correct dtypes, Breed1/Color1 as
        pandas Categorical with the fixed category sets applied.
    """

    # Normalising input to a DataFrame. A single dict -> one-row frame.
    # After this line, training and inference are IDENTICAL.
    if isinstance(raw, dict):
        df = pd.DataFrame([raw])
    else:
        df = raw.copy()

    # Loading the category sets if not supplied.
    if category_sets is None:
        category_sets = load_category_sets()

    # Adding derived features (desc_word_count, is_free).
    df = add_derived_features(df)

    # Applying categorical dtype with the FIXED category sets.
    # Any value not in the set (incl. unexpected breeds) becomes NaN, which XGBoost safely handles as missing.
    for col in CATEGORICAL_FEATURES:
        df[col] = pd.Categorical(df[col], categories=category_sets[col])

    # Selecting and ordering columns. 
    # Fails loudly if a required feature is missing.
    missing = [c for c in FEATURE_ORDER if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    return df[FEATURE_ORDER]


# === Schema Persistence (save with the model, load at inference) ===

def save_schema(path, category_sets: dict | None = None) -> None:
    """Persist the feature contract + category sets next to the model, so
    inference reproduces training exactly without re-reading the label CSVs."""

    if category_sets is None:
        category_sets = load_category_sets()
    schema = {
        "feature_order": FEATURE_ORDER,
        "categorical_features": CATEGORICAL_FEATURES,
        "category_sets": category_sets,
        "target": TARGET,
    }
    Path(path).write_text(json.dumps(schema, indent=2))

def load_schema(path) -> dict:
    """Load the saved schema. Returns a dict with feature_order, category_sets, etc."""

    return json.loads(Path(path).read_text())