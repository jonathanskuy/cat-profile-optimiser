"""
explain.py (SHAP-based per-listing explanations for Cat Profile Optimizer).

Wraps shap.TreeExplainer to turn one cat's prediction into a ranked list of
factor dicts (the contract recommend.py + the UI consume).

Impacts are expressed in approximate SCORE-POINTS (0-100 scale), not native
SHAP log-odds units, so they align with the score the user sees. This is a
local linear approximation: each feature's score-point impact is its share of
the total SHAP magnitude times the score's deviation from the baseline score.
Approximate because log-odds -> probability is nonlinear, but intuitive and
directly usable downstream.
"""


from pathlib import Path
import numpy as np
import pandas as pd
import shap
from preprocess import FEATURE_ORDER


# Labels for each feature
FEATURE_LABELS = {
    "Breed1":          "Breed",
    "Color1":          "Color",
    "Age":             "Age",
    "MaturitySize":    "Maturity size",
    "FurLength":       "Fur length",
    "Vaccinated":      "Vaccination status",
    "Dewormed":        "Deworming status",
    "Sterilized":      "Sterilization status",
    "Health":          "Health status",
    "Fee":             "Adoption fee",
    "PhotoAmt":        "Number of photos",
    "desc_word_count": "Description length",
    "is_free":         "Free adoption",
}


# === Build the Explainer (once, reused across predictions) ===
def build_explainer(model):
    """Construct a SHAP TreeExplainer for the fitted XGBoost model.

    Build this ONCE (e.g. at app startup) and reuse it for every cat.
    Constructing it is the expensive part; explaining one cat is cheap.

    Returns a shap.TreeExplainer.
    """

    return shap.TreeExplainer(model)


# === Explain Predictions ===
def explain_prediction(model, explainer, features: pd.DataFrame) -> list[dict]:
    """Explain ONE cat's score as a ranked list of factor dicts.

    Contract (consumed by recommend.py + the UI), sorted by absolute impact,
    most impactful first:
        {"feature": str, "label": str, "value": any,
         "impact": float (approx score-points), "direction": "positive"|"negative"}

    Impacts are approximate score-points (see module docstring): each feature's
    share of total SHAP magnitude, scaled to the score's deviation from the
    baseline score. Sums of impacts approximate (final_score - baseline_score).
    """

    if len(features) != 1:
        raise ValueError("explain_prediction handles one cat at a time")

    # Defining raw SHAP values (log-odds) for this cat
    shap_vals = np.array(explainer.shap_values(features))[0]

    # Converting to the 0-100 score space
    # baseline score    =   sigmoid(expected_value) * 100
    # final score       =   sigmoid(expected_value + sum(shap)) * 100
    base_logodds = float(explainer.expected_value)
    final_logodds = base_logodds + shap_vals.sum()
    baseline_score = _sigmoid(base_logodds) * 100
    final_score = _sigmoid(final_logodds) * 100
    score_delta = final_score - baseline_score

    # Distributing the score delta across features in proportion to |SHAP|
    total_mag = np.abs(shap_vals).sum()
    if total_mag == 0:
        impacts = np.zeros_like(shap_vals)
    else:
        # keep each feature's sign; scale magnitudes to sum to score_delta
        impacts = (shap_vals / total_mag) * abs(score_delta)
        # re-apply the sign of the overall delta correctly via shap sign already kept

    # 4. Building the ranked contract
    factors = []
    for feat, raw_shap, impact in zip(features.columns, shap_vals, impacts):
        factors.append({
            "feature": feat,
            "label": FEATURE_LABELS.get(feat, feat),
            "value": features.iloc[0][feat],
            "impact": round(float(impact), 1),
            "direction": "positive" if raw_shap >= 0 else "negative",
        })

    factors.sort(key=lambda f: abs(f["impact"]), reverse=True)
    
    return factors

def _sigmoid(x: float) -> float:
    """Logistic function: log-odds -> probability."""

    return 1.0 / (1.0 + np.exp(-x))


# === Global Feature Importance (across many cats) ===
def global_importance(explainer, features: pd.DataFrame) -> list[dict]:
    """Mean absolute SHAP value per feature across a set of cats.

    Shows which features drive the model overall (not for one cat). Useful for
    the UI's 'what matters most' view and as a sanity check that the model
    learned sensible patterns. Returns, sorted by importance (highest first):
        [{"feature": str, "label": str, "importance": float}, ...]

    Note: importance here is in native SHAP magnitude (log-odds), used only
    for RANKING features relative to each other.
    """
    shap_vals = np.array(explainer.shap_values(features))
    mean_abs = np.abs(shap_vals).mean(axis=0)

    importance = [
        {"feature": feat,
         "label": FEATURE_LABELS.get(feat, feat),
         "importance": round(float(val), 4)}
        for feat, val in zip(features.columns, mean_abs)
    ]
    importance.sort(key=lambda f: f["importance"], reverse=True)

    return importance