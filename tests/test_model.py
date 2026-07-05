"""Unit tests for model.py (scoring and persistence)."""


import pytest
import preprocess
import model as model_mod


SAMPLE_CAT = {
    "Age": 12, "Breed1": 243, "Color1": 1, "MaturitySize": 2, "FurLength": 1,
    "Vaccinated": 1, "Dewormed": 1, "Sterilized": 2, "Health": 1,
    "Fee": 0, "PhotoAmt": 3, "Description": "Friendly tabby, loves people.",
}


@pytest.fixture(scope="module")
def clf():
    """Load the committed model once for all tests in this module."""

    return model_mod.load_model()


def test_score_in_valid_range(clf):
    """The 0-100 score is always within bounds."""

    features = preprocess.preprocess(SAMPLE_CAT)
    score = model_mod.predict_score(clf, features)[0]
    assert 0.0 <= score <= 100.0


def test_score_is_deterministic(clf):
    """The same cat scores the same every time (no randomness at inference)."""

    features = preprocess.preprocess(SAMPLE_CAT)
    s1 = model_mod.predict_score(clf, features)[0]
    s2 = model_mod.predict_score(clf, features)[0]
    assert s1 == s2


def test_more_photos_not_worse(clf):
    """Sanity: more photos shouldn't lower the score for an otherwise
    identical cat (photos are a positive lever in the data)."""
    
    few = preprocess.preprocess({**SAMPLE_CAT, "PhotoAmt": 0})
    many = preprocess.preprocess({**SAMPLE_CAT, "PhotoAmt": 4})