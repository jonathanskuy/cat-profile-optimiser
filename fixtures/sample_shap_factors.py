"""
Mock SHAP output for developing src/recommend.py before the real model exists.

Each fixture is a list of factor dicts matching the agreed explain.py contract:
    {
        "feature":      str,    # raw feature name
        "label":        str,    # human-readable
        "value":        any,    # the cat's actual value for this feature
        "impact":       float,  # signed contribution to the 0-100 score
        "direction":    "positive" | "negative",
    }
Sorted by absolute impact, most impactful first.

These are HAND-MADE approximations grounded in the EDA, not real model output.
When the real explain.py lands, the shape stays the same; exact numbers will differ.

Use these to drive recommend.py + its tests. Key cases to get right:
  - CASE_WEAK_LISTING       -> should yield strong photo + description recs
  - CASE_STRONG_LISTING     -> should yield few/no recs (don't over-recommend)
  - CASE_STERILIZED_KITTEN  -> MUST NOT recommend changing sterilization (confound!)
  - CASE_UNSURE_STATUS      -> should recommend CONFIRMING status (transparency)
"""

# A poor listing: no photos, tiny description, unsure vaccination.
# Expected recs: strong "add photos", "expand description", "confirm vaccination".
CASE_WEAK_LISTING = [
    {"feature": "photo_count",      "label": "Number of photos",    "value": 0,         "impact": -15.0,"direction": "negative"},
    {"feature": "desc_word_count",  "label": "Description length",  "value": 6,         "impact": -8.0, "direction": "negative"},
    {"feature": "vaccinated",       "label": "Vaccination status",  "value": "Not sure","impact": -5.0, "direction": "negative"},
    {"feature": "age",              "label": "Age",                 "value": 18,        "impact": -6.0, "direction": "negative"},  # contextual: NO rec
    {"feature": "fur_length",       "label": "Fur length",          "value": "Short",   "impact": -2.0, "direction": "negative"},  # contextual: NO rec
]

# A strong listing: lots of photos, good description, status confirmed.
# Expected: few or no recommendations. Tests that recommend.py doesn't
# over-recommend (e.g. must NOT say "add more photos" to a 5-photo cat).
CASE_STRONG_LISTING = [
    {"feature": "photo_count",      "label": "Number of photos",    "value": 5,     "impact": 9.0,  "direction": "positive"},
    {"feature": "age",              "label": "Age",                 "value": 2,     "impact": 12.0, "direction": "positive"},  # kitten, contextual
    {"feature": "desc_word_count",  "label": "Description length",  "value": 140,   "impact": 5.0,  "direction": "positive"},
    {"feature": "sterilized",       "label": "Sterilization status","value": "Yes", "impact": 2.0,  "direction": "positive"},
    {"feature": "fee",              "label": "Adoption fee",        "value": 0,     "impact": 1.5,  "direction": "positive"},
]

# THE CONFOUND CASE: a young kitten that is NOT sterilized.
# Raw EDA would suggest "not sterilized = adopts faster", but that's because
# unsterilized cats are mostly kittens. recommend.py MUST NOT recommend
# changing sterilization status here (or anywhere). At most, if status were
# "Not sure", recommend confirming it. Here it's a known "No" on a kitten ->
# no sterilization rec at all.
CASE_STERILIZED_KITTEN = [
    {"feature": "age",              "label": "Age",                 "value": 3,     "impact": 14.0, "direction": "positive"},  # contextual
    {"feature": "photo_count",      "label": "Number of photos",    "value": 2,     "impact": 1.0,  "direction": "positive"},
    {"feature": "sterilized",       "label": "Sterilization status","value": "No",  "impact": 4.0,  "direction": "positive"},  # do NOT touch
    {"feature": "desc_word_count",  "label": "Description length",  "value": 80,    "impact": 0.5,  "direction": "positive"},
]

# Mid listing where multiple statuses are "Not sure".
# Expected: recommend CONFIRMING/stating the statuses (transparency framing),
# never recommend changing the underlying medical fact.
CASE_UNSURE_STATUS = [
    {"feature": "sterilized",   "label": "Sterilization status","value": "Not sure","impact": -6.0, "direction": "negative"},
    {"feature": "vaccinated",   "label": "Vaccination status",  "value": "Not sure","impact": -5.0, "direction": "negative"},
    {"feature": "photo_count",  "label": "Number of photos",    "value": 3,         "impact": 2.0,  "direction": "positive"},
    {"feature": "dewormed",     "label": "Deworming status",    "value": "Not sure","impact": -2.0, "direction": "negative"},
    {"feature": "health",       "label": "Health status",       "value": "Healthy", "impact": 3.0,  "direction": "positive"},
]

# A listing with a notably high fee (rare; 81% are free).
# Expected: at most a SOFT, conditional fee rec ("lower/waived fees adopt
# slightly faster, if your shelter has flexibility"). Not a headline rec.
CASE_HIGH_FEE = [
    {"feature": "photo_count",      "label": "Number of photos",    "value": 4,     "impact": 6.0,  "direction": "positive"},
    {"feature": "fee",              "label": "Adoption fee",        "value": 150,   "impact": -4.0, "direction": "negative"},
    {"feature": "desc_word_count",  "label": "Description length",  "value": 110,   "impact": 3.0,  "direction": "positive"},
    {"feature": "age",              "label": "Age",                 "value": 8,     "impact": -3.0, "direction": "negative"},  # contextual
]

# Convenience: all cases for iterating in tests.
ALL_CASES = {
    "weak_listing":     CASE_WEAK_LISTING,
    "strong_listing":   CASE_STRONG_LISTING,
    "sterilized_kitten":CASE_STERILIZED_KITTEN,
    "unsure_status":    CASE_UNSURE_STATUS,
    "high_fee":         CASE_HIGH_FEE,
}

# Which features are ACTIONABLE (may produce recommendations) vs CONTEXTUAL
# (explain the score only — never a recommendation). recommend.py should use
# a set like this so it never emits advice for things a shelter can't change.
ACTIONABLE_FEATURES = {"photo_count", "desc_word_count", "vaccinated",
                       "dewormed", "sterilized", "fee"}
CONTEXTUAL_FEATURES = {"age", "breed", "color", "fur_length",
                       "maturity_size", "health"}

if __name__ == "__main__":
    # Quick sanity print
    for name, factors in ALL_CASES.items():
        print(f"\n{name} ({len(factors)} factors):")
        for f in factors:
            tag = "ACTIONABLE" if f["feature"] in ACTIONABLE_FEATURES else "context"
            print(f"  {f['impact']:+5.1f}  {f['label']:<22} = {str(f['value']):<10} [{tag}]")