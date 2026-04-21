import streamlit as st

from ui_utils import disclaimer, footer, hero, setup_page, stat_card, step_card


setup_page("Home")

hero(
    "💧 AquaCheck",
    "Is Your Water Safe to Drink?",
    "Enter 9 simple chemical readings and get an instant AI-powered verdict. No technical knowledge needed — just your water report.",
)

# ── Stats row ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    stat_card("🎯", "80%", "Model Accuracy")
with c2:
    stat_card("🧪", "9", "Chemical Parameters")
with c3:
    stat_card("📊", "3,276", "Water Samples Trained On")
with c4:
    stat_card("⚡", "<1s", "Prediction Time")

st.write("")

# ── How it works ───────────────────────────────────────────────────────────
st.markdown(
    "<h3 style='text-align:center;margin-bottom:20px;'>How It Works</h3>",
    unsafe_allow_html=True,
)
s1, s2, s3 = st.columns(3)
with s1:
    step_card("📋", "1", "Enter Your Readings", "Go to Check My Water and fill in the 9 chemical values from your water quality report. All fields are pre-filled with safe typical values.")
with s2:
    step_card("🤖", "2", "AI Analyses Instantly", "Our XGBoost model — trained on thousands of real water samples — analyses the combination of all 9 parameters simultaneously.")
with s3:
    step_card("✅", "3", "Get Your Verdict", "See a clear Safe or Unsafe result with a confidence score. No jargon, no confusion — just a straight answer.")

st.write("")
st.markdown("---")

# ── Why trust it ───────────────────────────────────────────────────────────
st.markdown(
    "<h3 style='text-align:center;margin-bottom:20px;'>Why Trust AquaCheck?</h3>",
    unsafe_allow_html=True,
)
t1, t2, t3 = st.columns(3)
with t1:
    st.markdown(
        """
        <div style="background:#ffffff;border:1px solid #bfdbfe;border-radius:14px;padding:22px;text-align:center;box-shadow:0 4px 24px rgba(3,105,161,0.10);">
            <div style="font-size:2rem;margin-bottom:10px;">🌳</div>
            <div style="font-weight:800;color:#0b1f35;margin-bottom:8px;">XGBoost Model</div>
            <div style="font-size:0.88rem;color:#4a6580;line-height:1.6;">
                Gradient boosting — the gold standard for tabular data. Learns complex patterns across all 9 water parameters at once.
            </div>
        </div>
        """, unsafe_allow_html=True,
    )
with t2:
    st.markdown(
        """
        <div style="background:#ffffff;border:1px solid #bfdbfe;border-radius:14px;padding:22px;text-align:center;box-shadow:0 4px 24px rgba(3,105,161,0.10);">
            <div style="font-size:2rem;margin-bottom:10px;">📈</div>
            <div style="font-weight:800;color:#0b1f35;margin-bottom:8px;">80% Accurate</div>
            <div style="font-size:0.88rem;color:#4a6580;line-height:1.6;">
                Validated on real held-out water samples. That's 4 out of every 5 predictions correct — strong for this type of data.
            </div>
        </div>
        """, unsafe_allow_html=True,
    )
with t3:
    st.markdown(
        """
        <div style="background:#ffffff;border:1px solid #bfdbfe;border-radius:14px;padding:22px;text-align:center;box-shadow:0 4px 24px rgba(3,105,161,0.10);">
            <div style="font-size:2rem;margin-bottom:10px;">🔍</div>
            <div style="font-weight:800;color:#0b1f35;margin-bottom:8px;">Transparent</div>
            <div style="font-size:0.88rem;color:#4a6580;line-height:1.6;">
                Visit How It Works to see exactly which chemicals the model relies on most — no black box.
            </div>
        </div>
        """, unsafe_allow_html=True,
    )

st.write("")
disclaimer()
footer()
