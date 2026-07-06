# Cat Adoption Profile Optimiser

> Help shelters and cat owners write better adoption listings so cats get adopted faster.

Built for the **HackTheKitty** hackathon (June 24 - July 7, 2026).

A user enters a cat's details and instantly gets an **adoption-success score (0–100)**, a **SHAP explanation** of what's driving that score, **actionable recommendations** to improve the listing, an **AI-rewritten description**, and a **live "what-if" re-score** showing how the score changes as the listing improves.

**Live demo:** [Streamlit Cloud Link](https://cat-profile-optimiser.streamlit.app)
**Demo video:** _[link - add before submission]_

---

## The Problem

Cats with poorly constructed adoption listings (missing details, weak descriptions, too few photos) wait longer to be adopted, tying up scarce shelter capacity. Shelter staff are stretched thin and don't always know what makes a listing effective. This tool scores a listing and shows, specifically, what to improve.

> The score reflects **how well a listing is constructed**, based on historical adoption patterns. It does not rank which cats are "worth" adopting.

---

## Features

- **Adoption score (0–100)** from an XGBoost model.
- **Per-listing explanation** | a SHAP-based breakdown of what's helping vs hurting the score, shown as a diverging bar chart in score-points.
- **Actionable recommendations** | targeted, threshold-aware advice (only for things a shelter can change).
- **AI-improved description** | an LLM rewrite of the listing text, using only the facts provided.
- **What-if re-score** | interactive sliders that re-score the listing live, showing the score climb as you improve it.

---

## Tech Stack

- **Language:** Python 3.11
- **ML:** XGBoost (native categorical support), scikit-learn
- **Explainability:** SHAP (TreeExplainer)
- **Experiment tracking:** MLflow (training only)
- **Frontend:** Streamlit, Plotly (charts)
- **LLM:** _[Anthropic / OpenAI (specify), used by the description-rewrite module]_
- **Data:** PetFinder.my Adoption Prediction (Kaggle), scoped to cats
- **Version control:** Git / GitHub
- **Security scanning:** Aikido

---

## Architecture

```
User (browser)
   │
   ▼
Streamlit app (app.py) UI + orchestration
   │
   ├─► preprocess.py ─► XGBoost model (model.py) ─► adoption score (0–100)
   │                              │
   │                              ▼
   │                       explain.py (SHAP) ─► per-listing factor breakdown
   │
   ├─► recommend.py ─► SHAP-driven, threshold-aware recommendations
   │
   └─► llm.py ─► AI-rewritten description (uses only provided facts)
```

The ML engine (`preprocess`, `model`, `explain`) is separated from the UI and from the "intelligence layer" (`recommend`, `llm`). Training happens offline; the app loads a saved model + schema at startup and never trains at runtime. A single `preprocess()` path is shared by training and inference, so there is no train/inference skew.

---

## Prerequisites

- **Python 3.11** (a matching version is recommended for loading the saved model)
- **Conda** or **venv** for an isolated environment
- The repo includes the **trained model** (`models/`) and the small **label CSVs** (`data/raw/*_labels.csv`), so **you do not need the Kaggle dataset to run the app.** The full dataset is only needed to retrain (see below).

---

## Setup & Running (fresh clone → running app)

```bash
# 1. Clone
git clone https://github.com/jonathanskuy/cat-profile-optimiser.git
cd cat-profile-optimiser

# 2. Create an environment (conda example)
conda create -n catopt python=3.11 -y
conda activate catopt

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Only for the AI-rewrite feature) provide an LLM API key
#    Create a .env file in the project root:
#        LLM_API_KEY=your_key_here
#    The app runs fully without this.
#    The score, explanation, recommendations, and what-if re-score all work. Only the description-rewrite needs a key.

# 5. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. Enter a cat's details in the sidebar and click **Analyse listing**.

> **LLM key note (for judges):** the app is fully functional without an LLM key (every feature except the description rewrite works). To test the rewrite, supply a free/limited key in `.env` as above. _[Specify the provider and where to get a free key.]_

---

## Retraining the Model (optional)

The trained model is committed, so this is only needed to reproduce training from scratch.

```bash
# 1. Download the PetFinder.my Adoption Prediction dataset from Kaggle:
#    https://www.kaggle.com/competitions/petfinder-adoption-prediction
#    Place train.csv in data/raw/ (the label CSVs are already in the repo).

# 2. Open and run the notebooks in order:
#    notebooks/01_eda.ipynb         exploration, target definition
#    notebooks/02_modelling.ipynb   training, MLflow tracking, evaluation,
#                                   SHAP, and saving models/model.joblib + schema.json
```

---

## Project Structure

```
cat-profile-optimiser/
├── app.py                  # Streamlit entry point
├── src/
│   ├── preprocess.py       # shared feature engineering (train + inference)
│   ├── model.py            # train, predict (0-100 score), save/load
│   ├── explain.py          # SHAP wrapper (per-cat + global importance)
│   ├── recommend.py        # SHAP → recommendations
│   └── llm.py              # description rewriting
├── models/                 # committed: trained model + feature schema
├── data/raw/               # committed: small label CSVs only (train.csv gitignored)
├── notebooks/              # EDA + modelling
├── fixtures/               # mock data for developing/testing modules
├── tests/                  # unit tests (pytest)
├── documentation/          # project report + security report
│   ├── SECURITY.md
│   └── CatProfileOptimiserProjectReport.docx
├── requirements.txt        # runtime dependencies (deployed app)
├── requirements-dev.txt    # dev/training dependencies
├── README.md
└── .gitignore
```

---

## Model & Evaluation

The target is binarised from PetFinder's `AdoptionSpeed`: **"adopted within a month" (speed 0–2) vs "slower / not adopted" (3–4)**, a naturally balanced split (~57% / 43%).

| Metric | Value |
|---|---|
| ROC-AUC (5-fold CV) | ~0.68 ± 0.01 |
| Calibration | Well-calibrated raw probabilities (Brier ≈ 0.22); post-hoc calibration measured but not needed |

Adoption speed is inherently noisy (it depends on factors the data can't see, like who visits a shelter that week), so an AUC in this range reflects the honest predictability ceiling rather than a weak model. Hyperparameter tuning was tracked in MLflow and confirmed (via cross-validation) to be within noise, so the simplest strong config was chosen.

**Top drivers (global SHAP importance):** age (dominant), breed, photo count, description length. The actionable levers the tool targets (photos and description) are among the most influential features.

---

## Security

See [`documentation/SECURITY.md`](./documentation/SECURITY.md) | responsible data handling, secrets management, secure coding practices, dependency hygiene, and the Aikido scan report (findings triaged and resolved).

---

## Testing

The project has **15 unit tests** (pytest) covering the ML engine — the train/inference consistency guarantee, derived features and edge cases, score range and determinism, the model save/load round-trip, the SHAP explanation contract, and a regression test for the security fix. A full test matrix (automated + manual UI checks) is in the [project report](./documentation/CatProfileOptimiserProjectReport.docx).

Run the tests with:

​```bash
python -m pytest tests/
​```

---

## Limitations & Honest Notes

- The score reflects patterns in one dataset (PetFinder.my, primarily Malaysia) and may not generalise to all shelters/regions.
- It estimates listing quality, not an individual cat's adoptability.
- The model deliberately excludes anything a shelter can't act on from its recommendations (age, breed, etc. explain the score but never generate advice).
- `desc_word_count` and `Breed1`/`Color1` importance may be slightly inflated as high-cardinality features; since breed/color are contextual (never recommended on), this doesn't affect the advice given.

---

## Team

- **Marcellinus Jonathan Evanda Indarto** | data, model, SHAP, app, security, deployment
- **Felix Colin Lianto** | recommendations engine, LLM description rewriting

## License

This project is licensed under the MIT License (see the [`LICENSE`](./LICENSE) file for details).

## Acknowledgements

- PetFinder.my and the Kaggle competition organisers for the dataset
- HackTheKitty organisers