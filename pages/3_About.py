from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui_utils import disclaimer, footer, hero, info_card, setup_page


setup_page("About")

hero(
    "📖 About AquaCheck",
    "Clean Water Starts With Knowing What's In It",
    "AquaCheck uses AI to give you an instant estimate of whether water is safe to drink — based on 9 chemical measurements. Simple, fast, and transparent.",
)

# ── Global water context ───────────────────────────────────────────────────
st.markdown(
    "<h3 style='text-align:center;margin-bottom:6px;'>Why Water Safety Matters</h3>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;color:#4a6580;margin-bottom:20px;'>The scale of the global water safety problem.</p>",
    unsafe_allow_html=True,
)

w1, w2, w3, w4 = st.columns(4)
for col, (icon, val, lbl) in zip(
    [w1, w2, w3, w4],
    [
        ("🌍", "2B+",   "People lack safe drinking water"),
        ("💀", "485K",  "Deaths/year from contaminated water"),
        ("🚰", "1 in 3","People have no handwashing facility"),
        ("💧", "3 days","Max survival without water"),
    ],
):
    with col:
        st.markdown(
            f"""
            <div style="background:#ffffff;border:1px solid #bfdbfe;border-radius:14px;
                        padding:20px;text-align:center;box-shadow:0 4px 24px rgba(3,105,161,0.10);">
                <div style="font-size:1.8rem;margin-bottom:6px;">{icon}</div>
                <div style="font-size:1.6rem;font-weight:900;color:#0b1f35;line-height:1.1;">{val}</div>
                <div style="font-size:0.78rem;color:#4a6580;margin-top:4px;line-height:1.4;">{lbl}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")
st.markdown("---")

# ── What the app does ──────────────────────────────────────────────────────
st.markdown("<h3 style='margin-bottom:16px;'>What Does AquaCheck Do?</h3>", unsafe_allow_html=True)
c1, c2 = st.columns([1.1, 0.9])
with c1:
    st.markdown(
        """
        <div style="background:#ffffff;border:1px solid #bfdbfe;border-radius:14px;
                    padding:24px;box-shadow:0 4px 24px rgba(3,105,161,0.10);line-height:1.8;">
            <p style="color:#1e3a52;">
                You enter <strong>9 chemical measurements</strong> from a water quality report —
                things like pH, turbidity, and chloramine levels. AquaCheck's AI model analyses
                all 9 values together and tells you whether the water is likely
                <strong style="color:#059669;">safe</strong> or
                <strong style="color:#dc2626;">unsafe</strong> to drink.
            </p>
            <p style="color:#1e3a52;margin-top:12px;">
                The model was trained on <strong>3,276 real water samples</strong> and achieves
                <strong>80% accuracy</strong> — meaning it gets the right answer 4 out of every 5 times.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#0369a1,#0891b2);border-radius:14px;
                    padding:24px;color:#ffffff;box-shadow:0 8px 32px rgba(3,105,161,0.30);">
            <div style="font-size:1rem;font-weight:800;margin-bottom:14px;color:#ffffff;">⚡ Quick Facts</div>
            <div style="font-size:0.88rem;line-height:2;color:#e0f2fe;">
                🎯 &nbsp; 80% prediction accuracy<br>
                🧪 &nbsp; 9 chemical parameters<br>
                📊 &nbsp; 3,276 training samples<br>
                🌳 &nbsp; XGBoost algorithm<br>
                ⚡ &nbsp; Instant results<br>
                🔓 &nbsp; Fully transparent model
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
st.markdown("---")

# ── Who is it for ──────────────────────────────────────────────────────────
st.markdown("<h3 style='margin-bottom:16px;'>Who Is It For?</h3>", unsafe_allow_html=True)
i1, i2, i3 = st.columns(3)
with i1:
    info_card("👨‍👩‍👧", "Everyday Users", "Anyone who has received a water quality report and wants a quick, plain-English answer about whether their water is safe.")
with i2:
    info_card("🎓", "Students & Researchers", "Exploring water safety, machine learning, or public health? AquaCheck shows how AI can be applied to real-world problems.")
with i3:
    info_card("🏫", "Educators", "A live, interactive demo of a machine learning pipeline — from raw data to a working prediction app.")

st.write("")
disclaimer()
footer()
