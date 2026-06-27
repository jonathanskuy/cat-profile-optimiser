"""
model.py (train, predict, and persist the Cat Profile Optimizer model).

Pure, reusable functions. NO MLflow here, as experiment tracking lives in the
training notebook that calls these, so the inference path (app) stays clean
and doesn't import MLflow.

The model is a binary XGBoost classifier on the FEATURE_ORDER contract from
preprocess.py. The 0-100 score is predict_proba(adopted_fast) * 100.
"""


from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from preprocess import FEATURE_ORDER, TARGET, CATEGORICAL_FEATURES


# === Paths & Config ===

_HERE = Path(__file__).resolve().parent
MODELS_DIR = _HERE.parent / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
SCHEMA_PATH = MODELS_DIR / "schema.json"

RANDOM_SEED = 42
TEST_SIZE = 0.20

# Baseline XGBoost params. We'll tune from here and track runs in MLflow.
# enable_categorical=True + tree_method="hist" is REQUIRED for native categorical handling of Breed1/Color1.
DEFAULT_PARAMS = {
    "enable_categorical": True,
    "tree_method": "hist",
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "random_state": RANDOM_SEED,
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


# === Data splitting ===

def split_data(features: pd.DataFrame, target: pd.Series):
    """Stratified train/test split.

    Stratified on the target so the ~56.6/43.4 balance is preserved in both
    splits (important for honest evaluation on a balanced binary problem).

    === Parameters ===
    features    :   pd.DataFrame    # already preprocessed (FEATURE_ORDER columns)
    target      :   pd.Series       # the binary adopted_fast

    === Returns ===
    X_train, X_test, y_train, y_test
    """

    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_SEED,
    )


# === Training ===

def train_model(X_train: pd.DataFrame, y_train: pd.Series, params: dict | None = None) -> XGBClassifier:
    """Fit an XGBoost binary classifier on preprocessed features.

    Expects X_train to already be through preprocess() (i.e. Breed1/Color1
    are category dtype and columns are in FEATURE_ORDER) enable_categorical
    in the params is what lets XGBoost split on those categories natively.

    === Parameters ===
    X_train, y_train    :   training features (preprocessed) and target
    params              :   dict, optional  # overrides DEFAULT_PARAMS (for tuning runs)

    === Returns ===
    A fitted XGBClassifier.
    """

    model_params = {**DEFAULT_PARAMS, **(params or {})}
    model = XGBClassifier(**model_params)
    model.fit(X_train, y_train)
    return model


# === Inference ===

def predict_score(model, features: pd.DataFrame) -> np.ndarray:
    """Return the 0-100 adoption score(s).

    features must already be through preprocess() (FEATURE_ORDER columns,
    Breed1/Color1 as category dtype). Works for one cat or many.
    """

    proba = model.predict_proba(features)[:, 1]   # P(adopted_fast)
    return proba * 100


# === Persistence ===

def save_model(model, path=MODEL_PATH) -> None:
    """Persist the fitted model with joblib.

    We use joblib (not xgboost's native save_model) because xgboost 3.0.4's
    save_model path hits an _estimator_type bug. joblib pickles the whole
    fitted object and loads cleanly with the same pinned xgboost version.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path=MODEL_PATH):
    """Load a joblib-saved model. Must be loaded with a compatible
    xgboost version (see pinned requirements)."""

    return joblib.load(path)