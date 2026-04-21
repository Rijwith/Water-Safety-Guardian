from pathlib import Path
import pickle
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui_utils import confidence_bar, disclaimer, footer, hero, setup_page


MODEL_PATH = ROOT / "mymodel.pkl"
DATA_PATH  = ROOT / "water_potability.csv"

# Each feature: (display name, unit, safe range hint, plain description)
FEATURE_META = {
    "ph":               ("pH Level",          "",       "Safe: 6.5 – 8.5",   "How acidic or alkaline the water is."),
    "Hardness":         ("Hardness",           "mg/L",   "Typical: 60 – 180", "Dissolved calcium & magnesium minerals."),
    "Solids":           ("Dissolved Solids",   "ppm",    "Safe: < 500",        "Total particles dissolved in water."),
    "Chloramines":      ("Chloramines",        "mg/L",   "Safe: < 4",          "Disinfectant added during treatment."),
    "Sulfate":          ("Sulfate",            "mg/L",   "Safe: < 250",        "Naturally occurring sulfate levels."),
    "Conductivity":     ("Conductivity",       "μS/cm",  "Typical: 200 – 800","How well water conducts electricity."),
    "Organic_carbon":   ("Organic Carbon",     "ppm",    "Safe: < 2",          "Organic matter present in water."),
    "Trihalomethanes":  ("Trihalomethanes",    "μg/L",   "Safe: < 80",         "By-products from chlorination."),
    "Turbidity":        ("Turbidity",          "NTU",    "Safe: < 4",          "How cloudy or clear the water looks."),
}


@st.cache_resource
def load_model_artifact() -> dict:
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_feature_stats(features: tuple[str, ...]) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    stats = df[list(features)].describe().T
    stats["median"] = df[list(features)].median(numeric_only=True)
    return stats


setup_page("Check My Water")
artifact  = load_model_artifact()
model     = artifact["model"]
scaler    = artifact["scaler"]
threshold = artifact["threshold"]
features  = tuple(artifact["features"])
stats     = load_feature_stats(features)

hero(
    "🔬 Water Safety Check",
    "Enter Your Water Readings",
    "Fill in the values from your water quality report. Every field shows the safe range — hover the label for more detail. Then hit the button for your instant verdict.",
)

# ── Input section ──────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="input-section-header">
        <div class="input-section-title">💧 9 Chemical Parameters</div>
        <div class="input-section-sub">Fields are pre-filled with typical average values. Change only what you know from your report.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

values = {}
cols = st.columns(3)
for i, feature in enumerate(features):
    display_name, unit, safe_hint, desc = FEATURE_META.get(feature, (feature, "", "", ""))
    label = f"{display_name} ({unit})" if unit else display_name

    lower  = float(stats.loc[feature, "min"])
    upper  = float(stats.loc[feature, "max"])
    median = float(stats.loc[feature, "median"])
    step   = max((upper - lower) / 200, 0.01)
    if feature == "ph":
        lower, upper, step = 0.0, 14.0, 0.1

    with cols[i % 3]:
        values[feature] = st.number_input(
            label,
            min_value=lower,
            max_value=upper,
            value=median,
            step=step,
            help=f"{desc}  |  {safe_hint}",
            key=feature,
        )

input_df = pd.DataFrame([[values[f] for f in features]], columns=features)

st.write("")
predict = st.button("🚰 Check My Water Safety", use_container_width=True)

# ── Result ─────────────────────────────────────────────────────────────────
if predict:
    input_scaled = scaler.transform(input_df)
    probability  = float(model.predict_proba(input_scaled)[0][1])
    is_safe      = probability >= threshold

    if is_safe:
        st.markdown(
            f"""
            <div class="result-safe">
                <div class="result-icon">✅</div>
                <div class="result-title">Water Appears Safe to Drink</div>
                <div class="result-body">
                    The AI model analysed all 9 chemical parameters and estimates this water sample
                    is <strong>potable</strong> with <strong>{probability * 100:.0f}% confidence</strong>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-risk">
                <div class="result-icon">🚫</div>
                <div class="result-title">Water Does Not Appear Safe</div>
                <div class="result-body">
                    The AI model estimates this water sample is <strong>not potable</strong>.
                    The potability confidence is only <strong>{probability * 100:.0f}%</strong> —
                    below the safety threshold. Consider getting it tested professionally.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    confidence_bar(probability, is_safe)

    # Quick parameter summary
    with st.expander("📋 View the values you entered", expanded=False):
        display_df = pd.DataFrame({
            "Parameter": [FEATURE_META[f][0] for f in features],
            "Your Value": [f"{values[f]:.2f}" for f in features],
            "Unit": [FEATURE_META[f][1] for f in features],
            "Safe Range": [FEATURE_META[f][2] for f in features],
        })
        st.dataframe(display_df, hide_index=True, use_container_width=True)

    st.write("")
    disclaimer()

footer()
