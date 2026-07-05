"""Unit tests for explain.py (the SHAP explanation contract)."""


import pytest
import preprocess
import model as model_mod
import explain as explain_mod


SAMPLE_CAT = {
    "Age": 12, "Breed1": 243, "Color1": 1, "MaturitySize": 2, "FurLength": 1,
    "Vaccinated": 1, "Dewormed": 1, "Sterilized": 2, "Health": 1,
    "Fee": 0, "PhotoAmt": 3, "Description": "Friendly tabby, loves people.",
}


@pytest.fixture(scope="module")
def explainer_and_features():
    """Load model, build explainer, preprocess one cat (once for all tests)."""
    
    clf = model_mod.load_model()
    explainer = explain_mod.build_explainer(clf)
    features = preprocess.preprocess(SAMPLE_CAT)
    return clf, explainer, features


def test_returns_one_factor_per_feature(explainer_and_features):
    """Explanation has one factor per model feature."""

    clf, explainer, features = explainer_and_features
    factors = explain_mod.explain_prediction(clf, explainer, features)
    assert len(factors) == len(preprocess.FEATURE_ORDER)


def test_factor_contract_shape(explainer_and_features):
    """Every factor dict has the exact keys recommend.py depends on."""

    clf, explainer, features = explainer_and_features
    factors = explain_mod.explain_prediction(clf, explainer, features)
    for f in factors:
        assert set(f.keys()) == {"feature", "label", "value", "impact", "direction"}
        assert f["direction"] in ("positive", "negative")
        assert isinstance(f["impact"], float)


def test_factors_sorted_by_absolute_impact(explainer_and_features):
    """Factors are ranked most-impactful-first (what the UI/recommend rely on)."""

    clf, explainer, features = explainer_and_features
    factors = explain_mod.explain_prediction(clf, explainer, features)
    impacts = [abs(f["impact"]) for f in factors]
    assert impacts == sorted(impacts, reverse=True)


def test_direction_matches_impact_sign(explainer_and_features):
    """A factor's direction is consistent with the sign of its impact."""

    clf, explainer, features = explainer_and_features
    factors = explain_mod.explain_prediction(clf, explainer, features)
    for f in factors:
        if f["impact"] > 0:
            assert f["direction"] == "positive"
        elif f["impact"] < 0:
            assert f["direction"] == "negative"


def test_global_importance_ranks_features(explainer_and_features):
    """Global importance returns all features, sorted by importance."""
    
    clf, explainer, features = explainer_and_features
    importance = explain_mod.global_importance(explainer, features)
    assert len(importance) == len(preprocess.FEATURE_ORDER)
    vals = [f["importance"] for f in importance]
    assert vals == sorted(vals, reverse=True)