"""Unit tests for preprocess.py (the shared train/inference path)."""


import pandas as pd
import pytest
import preprocess


# A minimal valid cat, as the app's form would produce it.
SAMPLE_CAT = {
    "Age": 12, "Breed1": 243, "Color1": 1, "MaturitySize": 2, "FurLength": 1,
    "Vaccinated": 1, "Dewormed": 1, "Sterilized": 2, "Health": 1,
    "Fee": 0, "PhotoAmt": 3, "Description": "Friendly tabby, loves people.",
}


def test_dict_returns_feature_order():
    """A single cat dict produces exactly FEATURE_ORDER columns, in order."""

    out = preprocess.preprocess(SAMPLE_CAT)
    assert list(out.columns) == preprocess.FEATURE_ORDER
    assert len(out) == 1


def test_dict_and_dataframe_agree():
    """The SAME input as a dict vs a one-row DataFrame yields identical
    columns and shape. This is the train/inference skew guarantee."""

    from_dict = preprocess.preprocess(SAMPLE_CAT)
    from_df = preprocess.preprocess(pd.DataFrame([SAMPLE_CAT]))
    assert list(from_dict.columns) == list(from_df.columns)
    assert from_dict.shape == from_df.shape


def test_derived_features_computed():
    """desc_word_count and is_free are derived correctly."""

    out = preprocess.preprocess(SAMPLE_CAT)
    assert out.iloc[0]["desc_word_count"] == 4   # 4 words in the description
    assert out.iloc[0]["is_free"] == 1           # Fee == 0


def test_empty_description_is_zero_words():
    """A missing/empty description yields 0 words, not an error."""

    out = preprocess.preprocess({**SAMPLE_CAT, "Description": ""})
    assert out.iloc[0]["desc_word_count"] == 0


def test_paid_cat_is_not_free():
    """is_free flips to 0 when there's a fee."""

    out = preprocess.preprocess({**SAMPLE_CAT, "Fee": 50})
    assert out.iloc[0]["is_free"] == 0


def test_categorical_columns_are_category_dtype():
    """Breed1/Color1 come out as category dtype (needed for XGBoost)."""

    out = preprocess.preprocess(SAMPLE_CAT)
    for col in preprocess.CATEGORICAL_FEATURES:
        assert str(out[col].dtype) == "category"


def test_missing_feature_raises():
    """An incomplete cat (missing a required field) fails loudly rather
    than silently producing a wrong-shaped result."""

    incomplete = {k: v for k, v in SAMPLE_CAT.items() if k != "Age"}
    with pytest.raises((ValueError, KeyError)):
        preprocess.preprocess(incomplete)