# Contributing & Collaboration Guide

A two-person hackathon project (HackTheKitty 2026) built asynchronously across timezones. This doc records how the work was split and the interface contracts between modules, so each part integrates cleanly.

---

## Working agreement

- **All code written within June 24 – July 7, 2026** (hackathon rule). Committed in-window; git history reflects this.
- **Branch + PR workflow.** Feature work happens on a branch and is merged into `main`, reviewed across the timezone gap.
- **One owner per module.** Contributors edit different files; shared edits (e.g. `requirements.txt`) are coordinated first.
- **Interface contracts (below) are fixed and changed only by agreement.** They're the boundary between modules, so integration stays clean.

---

## Module ownership

| Module | Owner | Notes |
|---|---|---|
| `notebooks/` (EDA, feature eng, training, eval) | M. Jonathan E. I. | Critical path (sequential) |
| `src/preprocess.py` | M. Jonathan E. I. | Shared by train + inference; defines the feature schema |
| `src/model.py` | M. Jonathan E. I. | Load model, predict, produce 0–100 score |
| `src/explain.py` | M. Jonathan E. I. | SHAP wrapper (produces the contract below) |
| `app.py` | M. Jonathan E. I. | Streamlit app + orchestration |
| `tests/` | M. Jonathan E. I. | pytest unit tests for the ML engine |
| Security (`documentation/SECURITY.md` + Aikido) | M. Jonathan E. I. | Security scan + triage |
| README, project report, deployment | M. Jonathan E. I. | Documentation + Streamlit Cloud deploy |
| `src/recommend.py` | Felix C. L. | Maps SHAP output → actionable recommendations |
| `src/llm.py` | Felix C. L. | LLM description rewriting |

The ML engine (`preprocess`, `model`, `explain`) and app were kept on Jonathan's side so the critical path was never blocked by the timezone gap. Felix owns the "intelligence layer" (recommendations + LLM rewrite), which plugs into the app via the contracts below. The app is built with graceful fallbacks, so it runs whether or not the partner modules are present.

---

## Interface contracts

The shapes passed between modules. Build against these, not the other module's internals. These are the **real, final** contracts produced by the built engine.

### Preprocess (form/data → features)

```python
def preprocess(raw, category_sets=None) -> pd.DataFrame:
    """Takes a cat as a dict (form) or DataFrame (training); returns a
    DataFrame with columns in FEATURE_ORDER. Same path for train + inference."""

FEATURE_ORDER: list[str]   # the 13 ordered columns the model expects
```

The 13 features: `Breed1`, `Color1`, `Age`, `MaturitySize`, `FurLength`, `Vaccinated`, `Dewormed`, `Sterilized`, `Health`, `Fee`, `PhotoAmt`, `desc_word_count`, `is_free`. Model + explainer + schema are saved together in `models/` so inference can't drift from training.

### Score (model → app)

```python
def predict_score(model, features: pd.DataFrame) -> np.ndarray:
    """Returns the adoption score(s) as a float array in [0, 100]."""
```

### SHAP output (explain → recommend → app) | THE KEY CONTRACT

`explain.py`'s `explain_prediction()` returns a list of factor dicts, sorted by absolute impact, most impactful first. This is the real output:

```python
[
    {
        "feature": "Age",               # raw feature name (see the 13 above)
        "label": "Age",                 # human-readable
        "value": 12,                    # the cat's actual value
        "impact": -13.7,                # approx score-points (0-100 scale), signed
        "direction": "negative"         # "positive" | "negative"
    },
    ...
]
```

`recommend.py` consumes this and maps the negative contributors to actions:

```python
def recommend(shap_factors: list[dict]) -> list[dict]:
    """Prioritized recommendations, e.g.:
    [{ "priority": 1, "text": "Add at least 2 more photos",
       "linked_feature": "PhotoAmt" }, ...]
    Only recommend ACTIONABLE features (photos, description, Fee, and
    confirming Vaccinated/Dewormed/Sterilized status). NEVER recommend on
    contextual features (Age, Breed1, Color1, MaturitySize, FurLength, Health),
    and NEVER recommend changing a cat's medical status, only confirming it."""
```

A mock fixture (`fixtures/sample_shap_factors.py`) matches this shape for isolated development.

### LLM rewrite (llm → app)

```python
def rewrite_description(attributes: dict, original_description: str) -> str:
    """Returns an improved description (~100-200 words, the EDA sweet spot).
    GUARDRAIL: uses ONLY facts in attributes / original_description.
    Never invents health, vaccination, age, or temperament claims."""
```

API key comes from env var `LLM_API_KEY`, never hardcoded. Handle API errors gracefully. If the rewrite fails, the rest of the app still renders.

---

## Definition of done (per module)

- Runs without errors against the agreed contract
- No hardcoded secrets
- App still renders if the module fails (graceful fallback)