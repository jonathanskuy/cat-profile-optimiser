"""
app.py (Cat Adoption Profile Optimizer (Streamlit)).

Wires the ML engine (preprocess, model, explain) into a UI:
  cat attributes -> adoption score -> SHAP explanation -> recommendations
  -> AI-rewritten description -> before/after re-score.
"""


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / "src"))
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
try:
    import llm as llm_mod
except ImportError:
    llm_mod = None


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
    st.session_state["analyzed_cat"] = build_cat_dict()

# Show results if we have an analysed cat (from this click OR a previous one)
if "analyzed_cat" in st.session_state:
    cat = st.session_state["analyzed_cat"]

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


    # === Recommendations ===

    st.subheader("How to Improve this Listing")

    if recommend_mod is not None:
        try:
            recs = recommend_mod.recommend(factors)
            if recs:
                for r in recs:
                    st.markdown(f"**{r.get('priority', '•')}.** {r['text']}")
            else:
                st.caption("No specific improvements identified, this listing is in good shape.")
        except Exception as e:
            st.caption(f"Recommendations unavailable ({type(e).__name__}).")
    else:
        st.info("Recommendations module coming soon | this is where actionable tips will appear "
                "(e.g. add more photos, expand the description, confirm vaccination status).")
    

    # === AI-improved description ===

    st.subheader("Improved Description")

    original_desc = cat["Description"].strip()

    if llm_mod is not None:
        if not original_desc:
            st.caption("Add a description in the sidebar to get an AI-improved version.")
        else:
            with st.spinner("Rewriting the description..."):
                try:
                    improved = llm_mod.rewrite_description(cat, original_desc)
                    col_orig, col_new = st.columns(2)
                    with col_orig:
                        st.markdown("**Original**")
                        st.write(original_desc)
                    with col_new:
                        st.markdown("**AI-Improved**")
                        st.write(improved)
                except Exception as e:
                    st.caption(f"Rewrite unavailable ({type(e).__name__}).")
    else:
        st.info("Description rewriting coming soon | this is where an AI-improved, "
                   "warmer version of the listing text will appear.")
    

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
    st.info("Enter a cat's details in the sidebar and click **Analyze listing** to see its adoption score.") 