"""
app.py (Cat Adoption Profile Optimizer (Streamlit)).

Wires the ML engine (preprocess, model, explain) into a UI:
  cat attributes -> adoption score -> SHAP explanation -> recommendations
  -> AI-rewritten description (Gemini API, built directly into this file)
  -> before/after re-score.

Before running, set your Gemini API key in the terminal:
    export GEMINI_API_KEY="your-key-here"   (macOS/Linux)
    $env:GEMINI_API_KEY="your-key-here"     (Windows PowerShell)
Then:
    streamlit run app.py
"""

import os
import sys
import traceback
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

import requests
import pandas as pd
import streamlit as st
import preprocess
import model as model_mod
import explain as explain_mod
import plotly

try:
    import recommend as recommend_mod
except ImportError:
    recommend_mod = None


# === Page Config ===

st.set_page_config(
    page_title="Cat Adoption Profile Optimizer",
    page_icon="🐱",
    layout="wide",
)

st.markdown("""
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] { min-width: 360px; max-width: 360px; }
    </style>
""", unsafe_allow_html=True)


# === Gemini API (built directly into this file) ===

MATURITY_MAP = {1: "Small", 2: "Medium", 3: "Large", 4: "Extra Large"}
FUR_MAP = {1: "Short", 2: "Medium", 3: "Long"}
YES_NO_MAP = {1: "Yes", 2: "No", 3: "Not sure"}
HEALTH_MAP = {1: "Healthy", 2: "Minor injury", 3: "Serious injury"}


def get_gemini_api_key() -> str:
    """
    Read the Gemini API key from the environment. Never hardcode the key
    in source. Set it in your terminal before running `streamlit run app.py`:

        export GEMINI_API_KEY="your-key-here"

    See README.md for full instructions.
    """
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Set it in your terminal before "
            "running the app, e.g.:\n"
            '  export GEMINI_API_KEY="your-key-here"   (macOS/Linux)\n'
            '  $env:GEMINI_API_KEY="your-key-here"      (Windows PowerShell)\n'
            "See README.md for details."
        )
    return key


def build_gemini_prompt(cat: dict, original_desc: str) -> str:
    """Construct a prompt asking Gemini to rewrite an existing description."""
    maturity = MATURITY_MAP.get(cat.get("MaturitySize"), str(cat.get("MaturitySize")))
    fur = FUR_MAP.get(cat.get("FurLength"), str(cat.get("FurLength")))
    vaccinated = YES_NO_MAP.get(cat.get("Vaccinated"), str(cat.get("Vaccinated")))
    dewormed = YES_NO_MAP.get(cat.get("Dewormed"), str(cat.get("Dewormed")))
    sterilized = YES_NO_MAP.get(cat.get("Sterilized"), str(cat.get("Sterilized")))
    health = HEALTH_MAP.get(cat.get("Health"), str(cat.get("Health")))

    return (
        "Rewrite the following cat adoption listing description to be warmer, "
        "more engaging, and more likely to attract adopters, while staying "
        "truthful to the facts given.\n\n"
        f"Original description: \"{original_desc}\"\n\n"
        "Cat facts to stay consistent with:\n"
        f"- Age: {cat.get('Age')} months\n"
        f"- Size: {maturity}\n"
        f"- Fur length: {fur}\n"
        f"- Vaccinated: {vaccinated}\n"
        f"- Dewormed: {dewormed}\n"
        f"- Sterilized: {sterilized}\n"
        f"- Health: {health}\n"
        f"- Adoption fee: ${cat.get('Fee')}\n\n"
        "Return only the rewritten description, no preamble or labels."
    )


def call_gemini_api(prompt: str, api_url: str, api_key: str) -> str:
    """Call the Gemini API and return generated text."""
    url = f"{api_url}?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.9,
            "maxOutputTokens": 2048,
        },
    }
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    result = response.json()

    # Defensive: the API is expected to return a JSON object, but guard against
    # any unexpected top-level shape (e.g. a list, or None on an empty body)
    # so a malformed response raises a clear RuntimeError instead of an
    # AttributeError/TypeError from blindly calling .get()/["..."] on it.
    if not isinstance(result, dict):
        raise RuntimeError(f"Unexpected Gemini response shape (not an object): {str(result)[:300]}")

    # If the prompt itself was blocked (safety filters etc.), surface that clearly
    # rather than falling through to the generic "unexpected shape" error below.
    prompt_feedback = result.get("promptFeedback")
    if isinstance(prompt_feedback, dict) and prompt_feedback.get("blockReason"):
        raise RuntimeError(
            f"Gemini blocked the prompt (blockReason={prompt_feedback['blockReason']})."
        )

    candidates = result.get("candidates")
    if isinstance(candidates, list) and len(candidates) > 0:
        candidate = candidates[0] or {}
        # NOTE: when Gemini blocks/truncates a response (safety filter, RECITATION,
        # or hitting maxOutputTokens before producing text), "content" can be present
        # but explicitly null, e.g. {"content": null, "finishReason": "SAFETY"}.
        # candidate.get("content", {}) only applies its default when the key is
        # *missing*, not when it's null, so a bare ".get(...)" chain here raises
        # AttributeError: 'NoneType' object has no attribute 'get'. Guard with `or {}`.
        content = candidate.get("content") or {}
        parts = content.get("parts") or []
        text = "".join((p or {}).get("text", "") for p in parts).strip()
        if text:
            return text
        finish_reason = candidate.get("finishReason", "UNKNOWN")
        safety = candidate.get("safetyRatings")
        detail = f" safetyRatings={safety}" if safety else ""
        raise RuntimeError(f"Gemini returned no text (finishReason={finish_reason}).{detail}")

    raise RuntimeError(f"Unexpected Gemini response shape: {str(result)[:300]}")


def rewrite_description(cat: dict, original_desc: str) -> str:
    """Rewrite a cat's listing description using the Gemini API."""
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    api_url = os.getenv(
        "GEMINI_API_URL",
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent",
    )
    api_key = get_gemini_api_key()
    prompt = build_gemini_prompt(cat, original_desc)
    return call_gemini_api(prompt, api_url, api_key)


def build_gemini_recommendations_prompt(cat: dict, factors: list, rule_based_hints: list = None) -> str:
    """
    Construct a prompt asking Gemini for practical, condition-aware advice
    on making the listing more appealing (e.g. flagging that the cat isn't
    yet vaccinated/sterilized and suggesting how to handle that in the listing),
    informed by which SHAP factors are currently hurting the predicted score,
    and grounded by any rule-based hints from the local recommend module.
    """
    maturity = MATURITY_MAP.get(cat.get("MaturitySize"), str(cat.get("MaturitySize")))
    fur = FUR_MAP.get(cat.get("FurLength"), str(cat.get("FurLength")))
    vaccinated = YES_NO_MAP.get(cat.get("Vaccinated"), str(cat.get("Vaccinated")))
    dewormed = YES_NO_MAP.get(cat.get("Dewormed"), str(cat.get("Dewormed")))
    sterilized = YES_NO_MAP.get(cat.get("Sterilized"), str(cat.get("Sterilized")))
    health = HEALTH_MAP.get(cat.get("Health"), str(cat.get("Health")))

    # Surface only the factors currently dragging the score down, worst first,
    # so the advice stays grounded in what's actually hurting this listing.
    negative_factors = sorted(
        (f for f in factors if f.get("impact", 0) < 0),
        key=lambda f: f["impact"],
    )[:5]
    if negative_factors:
        factors_text = "\n".join(f"- {f['label']} ({f['impact']:+.1f} pts)" for f in negative_factors)
    else:
        factors_text = "- No significant negative factors identified."

    # Fold the local rule-based engine's output in as grounding signals.
    # These are hints for Gemini to build on/rewrite/prioritize, not a
    # separate list to be shown verbatim. Accepts either plain strings or
    # dicts with a "text" key, so it degrades gracefully if recommend_mod's
    # return shape doesn't match what's expected.
    hints_text = "- (none available)"
    if rule_based_hints:
        lines = []
        for h in rule_based_hints:
            if isinstance(h, dict):
                lines.append(str(h.get("text", h)))
            else:
                lines.append(str(h))
        if lines:
            hints_text = "\n".join(f"- {line}" for line in lines)

    return (
        "You are advising a foster/shelter volunteer on how to make a cat "
        "adoption listing more appealing to potential adopters, given the "
        "cat's current condition. Suggest 3-5 concrete, practical steps. "
        "Where vaccination, deworming, or sterilization status is unconfirmed "
        "('Not sure'), recommend CONFIRMING and clearly STATING the status in "
        "the listing — do NOT advise changing a cat's medical status (e.g. do "
        "not tell them to get the cat sterilized), as that may be inappropriate "
        "for the cat's age and is not the listing's problem to solve. Also address "
        "fee, photos, and description quality. "
        "Use the rule-based signals below as a starting point and grounding "
        "context, but rewrite them in your own words, prioritize what matters "
        "most, and expand with anything else relevant. Do not just restate "
        "them verbatim, and do not invent facts not given below.\n\n"
        "Cat's current condition:\n"
        f"- Age: {cat.get('Age')} months\n"
        f"- Size: {maturity}\n"
        f"- Fur length: {fur}\n"
        f"- Vaccinated: {vaccinated}\n"
        f"- Dewormed: {dewormed}\n"
        f"- Sterilized: {sterilized}\n"
        f"- Health: {health}\n"
        f"- Adoption fee: ${cat.get('Fee')}\n"
        f"- Number of photos: {cat.get('PhotoAmt')}\n\n"
        f"Rule-based signals (from internal scoring engine):\n{hints_text}\n\n"
        f"Factors currently hurting the predicted adoption score:\n{factors_text}\n\n"
        "Return only a markdown bullet list of recommendations, no preamble or labels."
    )


def generate_ai_recommendations(cat: dict, factors: list, rule_based_hints: list = None) -> str:
    """Generate personalized, condition-aware listing recommendations via Gemini,
    grounded by the local rule-based recommend module when available."""
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    api_url = os.getenv(
        "GEMINI_API_URL",
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent",
    )
    api_key = get_gemini_api_key()
    prompt = build_gemini_recommendations_prompt(cat, factors, rule_based_hints)
    return call_gemini_api(prompt, api_url, api_key)


# === Cached Loaders (run once, reused across interactions) ===

@st.cache_resource
def load_engine():
    """Load model, schema, explainer once. Cached for the app's lifetime."""

    clf = model_mod.load_model()
    schema = preprocess.load_schema(model_mod.SCHEMA_PATH)
    explainer = explain_mod.build_explainer(clf)
    return clf, schema, explainer

@st.cache_data
def load_label_maps():
    """Load breed/color name<->ID maps from the label CSVs. Cached."""

    breeds = pd.read_csv(preprocess.BREED_LABELS_PATH)
    colors = pd.read_csv(preprocess.COLOR_LABELS_PATH)

    cat_breeds = breeds[breeds["Type"] == 2]
    breed_name_to_id = dict(zip(cat_breeds["BreedName"], cat_breeds["BreedID"]))
    color_name_to_id = dict(zip(colors["ColorName"], colors["ColorID"]))

    return breed_name_to_id, color_name_to_id


# Loading everything

clf, schema, explainer = load_engine()
breed_name_to_id, color_name_to_id = load_label_maps()

st.title("Cat Adoption Profile Optimizer")
st.caption("Help your cat get adopted faster (score a listing, see what's holding it back, and improve it).")


# === Input Form (sidebar) ===

YES_NO_UNSURE = {"Yes": 1, "No": 2, "Not sure": 3}
MATURITY = {"Small": 1, "Medium": 2, "Large": 3, "Extra Large": 4}
FUR = {"Short": 1, "Medium": 2, "Long": 3}
HEALTH = {"Healthy": 1, "Minor injury": 2, "Serious injury": 3}

with st.sidebar:
    st.header("Cat Details")

    # Quick-demo button: loads and analyses the example cat in one click
    if st.button("Load example cat", use_container_width=True):
        st.session_state["analysed_cat"] = {
            "Age": 5, "Breed1": 265, "Color1": 1,
            "MaturitySize": 2, "FurLength": 2,
            "Vaccinated": 3, "Dewormed": 3, "Sterilized": 3,
            "Health": 1, "Fee": 0, "PhotoAmt": 1,
            "Description": "Very friendly yet shy cat",
        }

    with st.form("cat_form"):
        age = st.number_input("Age (months)", min_value=0, max_value=300, value=12)

        breed_name = st.selectbox("Breed", options=sorted(breed_name_to_id.keys()))
        color_name = st.selectbox("Primary Colour", options=sorted(color_name_to_id.keys()))

        maturity_name = st.selectbox("Maturity Size", options=list(MATURITY.keys()), index=1)
        fur_name = st.selectbox("Fur Length", options=list(FUR.keys()))

        vacc_name = st.selectbox("Vaccinated", options=list(YES_NO_UNSURE.keys()))
        deworm_name = st.selectbox("Dewormed", options=list(YES_NO_UNSURE.keys()))
        ster_name = st.selectbox("Sterilized", options=list(YES_NO_UNSURE.keys()))
        health_name = st.selectbox("Health", options=list(HEALTH.keys()))

        fee = st.number_input("Adoption Fee", min_value=0, max_value=1000, value=0)
        photos = st.number_input("Number of Photos", min_value=0, max_value=30, value=3)

        description = st.text_area("Listing Description",
                                   value="", height=120,
                                   placeholder="Describe the cat's personality, habits, story, etc.")

        submitted = st.form_submit_button("Analyse Listing", use_container_width=True)


# === Cat Dict Assembly upon Submission ===

def build_cat_dict():
    """Convert form inputs into the raw dict preprocess() expects."""

    return {
        "Age": age,
        "Breed1": breed_name_to_id[breed_name],
        "Color1": color_name_to_id[color_name],
        "MaturitySize": MATURITY[maturity_name],
        "FurLength": FUR[fur_name],
        "Vaccinated": YES_NO_UNSURE[vacc_name],
        "Dewormed": YES_NO_UNSURE[deworm_name],
        "Sterilized": YES_NO_UNSURE[ster_name],
        "Health": HEALTH[health_name],
        "Fee": fee,
        "PhotoAmt": photos,
        "Description": description,
    }


# === Main Results Panel ===

# When 'Analyse Listing' is clicked, store the cat so results persist across slider tweaks
if submitted:
    st.session_state["analysed_cat"] = build_cat_dict()

# Show results if we have an analysed cat (from this click OR a previous one)
if "analysed_cat" in st.session_state:
    cat = st.session_state["analysed_cat"]

    features = preprocess.preprocess(cat, category_sets=schema["category_sets"])
    score = float(model_mod.predict_score(clf, features)[0])
    factors = explain_mod.explain_prediction(clf, explainer, features)

    st.subheader("Adoption Score")
    st.metric(label="Predicted Adoption Success", value=f"{score:.0f} / 100")
    st.progress(score / 100)

    if score >= 65:
        st.success("Strong Listing (this cat is well-positioned for a fast adoption)")
    elif score >= 45:
        st.info("Decent Listing (a few improvements could help it stand out)")
    else:
        st.warning("This listing may be holding the cat back. See what to improve below.")


    # === SHAP Explanations ===

    st.subheader("What's Affecting this Score")

    import plotly.graph_objects as go

    # Keep only factors that meaningfully moved the score, take the top ~8 by magnitude
    significant = [f for f in factors if abs(f["impact"]) >= 0.5][:8]
    # Reverse so the biggest impact sits at the TOP of the horizontal chart
    significant = significant[::-1]

    labels = [f["label"] for f in significant]
    impacts = [f["impact"] for f in significant]
    colors = ["#B01212" if v < 0 else "#1AAB18" for v in impacts]

    fig = go.Figure(go.Bar(
        x=impacts,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.0f}" for v in impacts],
        textposition="outside",
        hovertemplate="%{y}: %{x:+.1f} pts<extra></extra>",
    ))
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Impact on score (points)",
        yaxis_title=None,
        showlegend=False,
    )
    fig.add_vline(x=0, line_width=1, line_color="gray")

    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔴 Holding the Score Back   |   🟢 Helping the Score")


    # === Recommendations (single section, powered by Gemini, grounded by
    # the local rule-based recommend module when it's available) ===

    st.subheader("How to Improve this Listing")

    # Gather rule-based hints first. These feed INTO the Gemini prompt as
    # grounding context rather than being displayed on their own — so if
    # recommend_mod is missing or throws, we just fall back to factors-only
    # grounding instead of showing a dead-end "unavailable" message.
    rule_based_hints = []
    if recommend_mod is not None:
        try:
            rule_based_hints = recommend_mod.recommend(factors) or []
        except Exception:
            rule_based_hints = []

    with st.spinner("Generating personalized suggestions..."):
        try:
            ai_recs = generate_ai_recommendations(cat, factors, rule_based_hints)
            st.markdown(ai_recs)
        except RuntimeError as e:
            # Raised by get_gemini_api_key() when GEMINI_API_KEY isn't set,
            # or by call_gemini_api() when Gemini blocks/returns no usable text.
            st.error(str(e))
        except Exception as e:
            st.caption(f"Recommendations unavailable ({type(e).__name__}: {e}).")
            with st.expander("Show error details"):
                st.code(traceback.format_exc())


    # === AI-improved description (Gemini, called directly from this file) ===

    st.subheader("Improved Description")

    original_desc = cat["Description"].strip()

    if not original_desc:
        st.caption("Add a description in the sidebar to get an AI-improved version.")
    else:
        with st.spinner("Rewriting the description..."):
            try:
                improved = rewrite_description(cat, original_desc)
                col_orig, col_new = st.columns(2)
                with col_orig:
                    st.markdown("**Original**")
                    st.write(original_desc)
                with col_new:
                    st.markdown("**AI-Improved**")
                    st.write(improved)
            except RuntimeError as e:
                # Raised by get_gemini_api_key() when GEMINI_API_KEY isn't set,
                # or by call_gemini_api() when Gemini blocks/returns no usable text.
                st.error(str(e))
            except Exception as e:
                st.caption(f"Rewrite unavailable ({type(e).__name__}: {e}).")
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())


    # === What-If (improve the listing and re-score) ===

    st.divider()
    st.subheader("What if You Improved the Listing?")
    st.caption("Adjust the fields below to see how the score could change.")

    col_a, col_b = st.columns(2)
    with col_a:
        new_photos = st.slider("Number of Photos", 0, 12,
                               value=int(cat["PhotoAmt"]))
    with col_b:
        new_desc_words = st.slider("Description Length (words)", 0, 250,
                                   value=len(str(cat["Description"]).split()))

    # Build a modified cat listing (same as original but with the what-if values)
    # The description length is simulated by padding to the target word count,
    # since preprocess derives desc_word_count from the text
    whatif_cat = dict(cat)
    whatif_cat["PhotoAmt"] = new_photos
    whatif_cat["Description"] = " ".join(["word"] * new_desc_words)  # length-only proxy

    whatif_features = preprocess.preprocess(whatif_cat, category_sets=schema["category_sets"])
    whatif_score = float(model_mod.predict_score(clf, whatif_features)[0])
    delta = whatif_score - score

    m1, m2 = st.columns(2)
    m1.metric("Current score", f"{score:.0f} / 100")
    m2.metric("Improved score", f"{whatif_score:.0f} / 100", delta=f"{delta:+.0f}")
    st.progress(whatif_score / 100)

else:
    st.info("Enter a cat's details in the sidebar and click **Analyse listing** to see its adoption score.")