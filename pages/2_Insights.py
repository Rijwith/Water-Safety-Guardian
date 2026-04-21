from pathlib import Path
import pickle
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui_utils import feature_importance_bars, footer, hero, setup_page


MODEL_PATH = ROOT / "models" / "mymodel.pkl"
PLOTS_DIR  = ROOT / "outputs_xgb" / "plots"

FEATURE_PLAIN = {
    "ph":              ("pH Level",         "🧪", "Measures acidity. Too high or too low makes water unsafe and affects how other chemicals behave."),
    "Hardness":        ("Hardness",         "🪨", "High mineral content. Not directly harmful but affects taste and pipe scaling."),
    "Solids":          ("Dissolved Solids", "🔬", "Total particles in water. Very high levels can indicate contamination."),
    "Chloramines":     ("Chloramines",      "💊", "Disinfectant that kills bacteria. Too much can form harmful by-products."),
    "Sulfate":         ("Sulfate",          "⚗️", "Naturally occurring. High levels can cause a bitter taste and digestive issues."),
    "Conductivity":    ("Conductivity",     "⚡", "Reflects how many ions are dissolved. Very high values suggest contamination."),
    "Organic_carbon":  ("Organic Carbon",   "🌿", "Organic matter from soil and plants. High levels can indicate pollution."),
    "Trihalomethanes": ("Trihalomethanes",  "☣️", "Chlorination by-products. Long-term exposure above safe limits is a health risk."),
    "Turbidity":       ("Turbidity",        "🌫️", "Cloudiness from suspended particles. High turbidity can harbour bacteria."),
}


@st.cache_resource
def load_artifact() -> dict:
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def importance_frame() -> pd.DataFrame:
    artifact = load_artifact()
    model    = artifact["model"]
    features = artifact["features"]
    rows = []
    for f, imp in zip(features, model.feature_importances_):
        name, icon, desc = FEATURE_PLAIN.get(f, (f, "🔹", ""))
        rows.append({"What it measures": f"{icon} {name}", "Description": desc, "Importance": imp, "raw": f})
    return pd.DataFrame(rows).sort_values("Importance", ascending=False).reset_index(drop=True)


setup_page("How It Works")

hero(
    "🧠 How It Works",
    "What Makes Water Safe or Unsafe?",
    "The AI examines 9 chemical properties together. Here's what it focuses on most — and what each measurement actually means for your health.",
)

importance = importance_frame()

# ── Feature importance ─────────────────────────────────────────────────────
st.markdown(
    """
    <div style="background:#ffffff;border:1px solid #bfdbfe;border-radius:14px;padding:24px 28px;margin-bottom:24px;box-shadow:0 4px 24px rgba(3,105,161,0.10);">
        <div style="font-size:1.1rem;font-weight:800;color:#0b1f35;margin-bottom:4px;">🔑 What the AI Pays Most Attention To</div>
        <div style="font-size:0.85rem;color:#4a6580;margin-bottom:20px;">Longer bar = stronger influence on the safety prediction.</div>
    """,
    unsafe_allow_html=True,
)
feature_importance_bars(importance)
st.markdown("</div>", unsafe_allow_html=True)

# ── Feature explainer cards ────────────────────────────────────────────────
st.markdown(
    "<h3 style='margin-bottom:16px;'>📋 What Each Parameter Means</h3>",
    unsafe_allow_html=True,
)

rows = [importance.iloc[i:i+3] for i in range(0, len(importance), 3)]
for row_df in rows:
    cols = st.columns(3)
    for col, (_, feat_row) in zip(cols, row_df.iterrows()):
        with col:
            st.markdown(
                f"""
                <div style="background:#ffffff;border:1px solid #bfdbfe;border-radius:12px;
                            padding:18px;margin-bottom:12px;box-shadow:0 2px 12px rgba(3,105,161,0.08);">
                    <div style="font-size:1rem;font-weight:800;color:#0b1f35;margin-bottom:6px;">
                        {feat_row['What it measures']}
                    </div>
                    <div style="font-size:0.83rem;color:#4a6580;line-height:1.55;">
                        {feat_row['Description']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ── ROC curve ─────────────────────────────────────────────────────────────
roc_path = PLOTS_DIR / "xgboost_roc_curve.png"
if roc_path.exists():
    st.markdown("---")
    st.markdown(
        """
        <div style="margin-bottom:10px;">
            <div style="font-size:1.1rem;font-weight:800;color:#0b1f35;">📈 How Reliable Is the Model?</div>
            <div style="font-size:0.85rem;color:#4a6580;margin-top:4px;">
                The curve below shows how well the model separates safe from unsafe water.
                The closer it hugs the top-left corner, the better. A random guess would follow the diagonal line.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.image(str(roc_path), use_container_width=True)

footer()
