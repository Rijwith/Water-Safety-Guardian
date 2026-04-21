from __future__ import annotations

import json
import random
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
XGB_RESULTS_PATH = ROOT / "outputs_xgb" / "xgb_results.json"

WATER_FACTS = [
    "💧 About 71% of the Earth's surface is covered by water, but only 3% is fresh water.",
    "🚰 Over 2 billion people worldwide lack access to safe drinking water.",
    "⚗️ Water is the only natural substance found in all three states — solid, liquid, and gas.",
    "🧪 The WHO recommends a pH between 6.5 and 8.5 for safe drinking water.",
    "🌍 A person can survive about 3 weeks without food, but only 3 days without water.",
    "💡 It takes about 140 litres of water to produce a single cup of coffee.",
    "🔬 Turbidity above 4 NTU is considered unsafe for drinking by WHO standards.",
    "📊 Chloramines are used to disinfect water — safe levels are below 4 mg/L.",
]


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_results() -> dict:
    return read_json(XGB_RESULTS_PATH)


def setup_page(title: str) -> None:
    st.set_page_config(
        page_title=f"{title} | AquaCheck",
        page_icon="💧",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    render_sidebar()


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        :root {
            --ink:      #0b1f35;
            --text:     #1e3a52;
            --muted:    #4a6580;
            --blue:     #0369a1;
            --teal:     #0891b2;
            --green:    #059669;
            --red:      #dc2626;
            --amber:    #d97706;
            --safe-bg:  #ecfdf5;
            --safe-bd:  #6ee7b7;
            --risk-bg:  #fef2f2;
            --risk-bd:  #fca5a5;
            --panel:    #ffffff;
            --line:     #bfdbfe;
            --page:     #f0f9ff;
            --card-shadow: 0 4px 24px rgba(3,105,161,0.10);
        }

        html, body, [data-testid="stAppViewContainer"] {
            background: var(--page) !important;
            font-family: 'Inter', sans-serif;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        h1,h2,h3,h4 { color: var(--ink); font-weight: 800; font-family: 'Inter', sans-serif; }
        p, li, label { color: var(--text); font-family: 'Inter', sans-serif; }

        /* Hide Streamlit's auto-generated top nav links */
        [data-testid="stSidebarNav"] {
            display: none !important;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0c2d4e 0%, #0a3d62 60%, #0b4f6c 100%) !important;
            border-right: none;
        }
        [data-testid="stSidebar"] * { color: #e0f2fe !important; }
        [data-testid="stSidebar"] .sidebar-brand {
            font-size: 1.5rem;
            font-weight: 900;
            color: #ffffff !important;
            letter-spacing: -0.5px;
        }
        [data-testid="stSidebar"] .sidebar-tagline {
            font-size: 0.78rem;
            color: #7dd3fc !important;
            margin-top: -4px;
            margin-bottom: 16px;
        }
        [data-testid="stSidebar"] .sidebar-acc-box {
            background: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 12px;
            padding: 14px 16px;
            margin-bottom: 16px;
            text-align: center;
        }
        [data-testid="stSidebar"] .sidebar-acc-label {
            font-size: 0.72rem;
            color: #7dd3fc !important;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        [data-testid="stSidebar"] .sidebar-acc-value {
            font-size: 2rem;
            font-weight: 900;
            color: #ffffff !important;
            line-height: 1.1;
        }
        [data-testid="stSidebar"] .sidebar-acc-sub {
            font-size: 0.75rem;
            color: #bae6fd !important;
        }
        [data-testid="stSidebar"] .nav-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 12px;
            border-radius: 8px;
            margin-bottom: 4px;
            font-size: 0.9rem;
            font-weight: 600;
            color: #e0f2fe !important;
            transition: background 0.2s;
        }
        [data-testid="stSidebar"] .nav-item:hover { background: rgba(255,255,255,0.12); }
        [data-testid="stSidebar"] .fact-box {
            background: rgba(14,165,233,0.15);
            border-left: 3px solid #38bdf8;
            border-radius: 0 8px 8px 0;
            padding: 10px 12px;
            font-size: 0.78rem;
            color: #bae6fd !important;
            margin-top: 16px;
            line-height: 1.5;
        }

        /* ── HERO ── */
        .app-hero {
            background: linear-gradient(135deg, #0369a1 0%, #0891b2 50%, #06b6d4 100%);
            border-radius: 16px;
            padding: 36px 40px;
            margin-bottom: 28px;
            position: relative;
            overflow: hidden;
            box-shadow: 0 20px 60px rgba(3,105,161,0.30);
        }
        .app-hero::before {
            content: "💧";
            position: absolute;
            right: 40px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 7rem;
            opacity: 0.15;
        }
        .eyebrow {
            color: #bae6fd;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        .hero-title {
            font-size: 2.6rem;
            line-height: 1.1;
            font-weight: 900;
            color: #ffffff;
            margin: 0 0 12px;
            max-width: 700px;
        }
        .hero-copy {
            color: #e0f2fe;
            font-size: 1.05rem;
            max-width: 680px;
            margin: 0;
            line-height: 1.6;
        }

        /* ── STEP CARDS (home page) ── */
        .step-card {
            background: #ffffff;
            border: 1px solid #bfdbfe;
            border-radius: 14px;
            padding: 24px 20px;
            text-align: center;
            box-shadow: var(--card-shadow);
            height: 100%;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .step-card:hover { transform: translateY(-4px); box-shadow: 0 12px 36px rgba(3,105,161,0.16); }
        .step-icon { font-size: 2.4rem; margin-bottom: 12px; }
        .step-num {
            display: inline-block;
            background: #0369a1;
            color: #ffffff;
            font-size: 0.7rem;
            font-weight: 800;
            border-radius: 999px;
            padding: 2px 9px;
            margin-bottom: 10px;
            letter-spacing: 0.06em;
        }
        .step-title { font-size: 1.05rem; font-weight: 800; color: var(--ink); margin-bottom: 8px; }
        .step-body { font-size: 0.88rem; color: var(--muted); line-height: 1.55; }

        /* ── STAT CARDS ── */
        .stat-card {
            background: #ffffff;
            border: 1px solid #bfdbfe;
            border-radius: 14px;
            padding: 20px;
            box-shadow: var(--card-shadow);
            text-align: center;
        }
        .stat-icon { font-size: 1.8rem; margin-bottom: 6px; }
        .stat-value { font-size: 1.9rem; font-weight: 900; color: var(--ink); line-height: 1.1; }
        .stat-label { font-size: 0.78rem; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 0.07em; margin-top: 4px; }

        /* ── INPUT SECTION ── */
        .input-section-header {
            background: linear-gradient(90deg, #f0f9ff, #e0f2fe);
            border: 1px solid #bae6fd;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 20px;
        }
        .input-section-title { font-size: 1.1rem; font-weight: 800; color: var(--ink); margin: 0 0 4px; }
        .input-section-sub { font-size: 0.85rem; color: var(--muted); margin: 0; }

        /* ── PARAMETER CARD ── */
        .param-card {
            background: #ffffff;
            border: 1.5px solid #e0f2fe;
            border-radius: 12px;
            padding: 14px 16px 10px;
            margin-bottom: 12px;
            box-shadow: 0 2px 8px rgba(3,105,161,0.06);
            transition: border-color 0.2s;
        }
        .param-card:hover { border-color: #7dd3fc; }
        .param-name { font-size: 0.78rem; font-weight: 700; color: var(--blue); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 2px; }
        .param-desc { font-size: 0.78rem; color: var(--muted); margin-bottom: 8px; line-height: 1.4; }

        /* ── RESULT PANELS ── */
        .result-safe {
            background: linear-gradient(135deg, #ecfdf5, #d1fae5);
            border: 2px solid var(--safe-bd);
            border-radius: 16px;
            padding: 28px 32px;
            margin: 16px 0;
            box-shadow: 0 8px 32px rgba(5,150,105,0.15);
            animation: fadeInUp 0.4s ease;
        }
        .result-risk {
            background: linear-gradient(135deg, #fef2f2, #fee2e2);
            border: 2px solid var(--risk-bd);
            border-radius: 16px;
            padding: 28px 32px;
            margin: 16px 0;
            box-shadow: 0 8px 32px rgba(220,38,38,0.15);
            animation: fadeInUp 0.4s ease;
        }
        .result-icon { font-size: 3rem; margin-bottom: 10px; }
        .result-title { font-size: 1.7rem; font-weight: 900; margin-bottom: 8px; }
        .result-safe .result-title { color: #065f46; }
        .result-risk .result-title { color: #991b1b; }
        .result-body { font-size: 1rem; line-height: 1.6; }
        .result-safe .result-body { color: #064e3b; }
        .result-risk .result-body { color: #7f1d1d; }

        /* ── CONFIDENCE BAR ── */
        .conf-bar-wrap {
            background: #e0f2fe;
            border-radius: 999px;
            height: 14px;
            overflow: hidden;
            margin: 10px 0 6px;
        }
        .conf-bar-fill {
            height: 100%;
            border-radius: 999px;
            transition: width 0.6s ease;
        }
        .conf-safe  { background: linear-gradient(90deg, #34d399, #059669); }
        .conf-risk  { background: linear-gradient(90deg, #f87171, #dc2626); }
        .conf-label { font-size: 0.82rem; color: var(--muted); text-align: right; }

        /* ── FEATURE IMPORTANCE BARS ── */
        .feat-row {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 10px;
        }
        .feat-name { font-size: 0.85rem; font-weight: 600; color: var(--ink); width: 160px; flex-shrink: 0; }
        .feat-bar-bg { flex: 1; background: #e0f2fe; border-radius: 999px; height: 10px; overflow: hidden; }
        .feat-bar-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #0891b2, #0369a1); }
        .feat-pct { font-size: 0.78rem; color: var(--muted); width: 40px; text-align: right; flex-shrink: 0; }

        /* ── INFO CARDS (about page) ── */
        .info-card {
            background: #ffffff;
            border: 1px solid #bfdbfe;
            border-radius: 14px;
            padding: 22px;
            box-shadow: var(--card-shadow);
            height: 100%;
        }
        .info-card-icon { font-size: 2rem; margin-bottom: 10px; }
        .info-card-title { font-size: 1rem; font-weight: 800; color: var(--ink); margin-bottom: 8px; }
        .info-card-body { font-size: 0.88rem; color: var(--muted); line-height: 1.6; }

        /* ── DISCLAIMER ── */
        .disclaimer {
            background: #fffbeb;
            border: 1px solid #fcd34d;
            border-radius: 12px;
            padding: 14px 18px;
            font-size: 0.85rem;
            color: #78350f;
            line-height: 1.6;
        }

        /* ── FOOTER ── */
        .footer {
            border-top: 1px solid #bfdbfe;
            color: var(--muted);
            font-size: 0.82rem;
            margin-top: 40px;
            padding-top: 18px;
            text-align: center;
        }

        /* ── STREAMLIT OVERRIDES ── */
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #bfdbfe;
            border-radius: 12px;
            padding: 14px 16px;
            box-shadow: var(--card-shadow);
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] p,
        div[data-testid="stMetric"] div { color: var(--ink); }

        div[data-baseweb="input"] input {
            color: var(--ink);
            background: #ffffff;
            border-radius: 8px;
        }
        div[data-baseweb="tab-list"] button { color: var(--muted); font-weight: 700; }
        div[data-baseweb="tab-list"] button[aria-selected="true"] { color: var(--blue); }

        .stButton > button {
            border-radius: 10px;
            min-height: 48px;
            font-size: 1rem;
            font-weight: 800;
            background: linear-gradient(135deg, #0369a1, #0891b2);
            border: none;
            color: #ffffff;
            box-shadow: 0 4px 16px rgba(3,105,161,0.30);
            transition: all 0.2s;
            letter-spacing: 0.02em;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #0284c7, #06b6d4);
            box-shadow: 0 6px 24px rgba(3,105,161,0.40);
            transform: translateY(-1px);
            color: #ffffff;
        }
        .stButton > button:active { transform: translateY(0); color: #ffffff; }

        [data-testid="stDataFrame"], [data-testid="stTable"] {
            background: #ffffff;
            border-radius: 10px;
        }

        div[data-testid="stExpander"] {
            background: #ffffff;
            border: 1px solid #bfdbfe;
            border-radius: 10px;
        }

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(16px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    xgb_results = load_results()
    acc = xgb_results["accuracy"] * 100
    fact = random.choice(WATER_FACTS)

    # ── 1. Logo at very top ──
    st.sidebar.markdown(
        """
        <div style="padding:8px 0 16px;border-bottom:1px solid rgba(255,255,255,0.12);margin-bottom:16px;">
        <svg viewBox="0 0 200 60" xmlns="http://www.w3.org/2000/svg" style="width:85%;max-width:190px;">
          <defs>
            <linearGradient id="sdropG" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="#7dd3fc"/>
              <stop offset="100%" stop-color="#0891b2"/>
            </linearGradient>
            <filter id="sglow">
              <feGaussianBlur stdDeviation="1.5" result="blur"/>
              <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
          </defs>
          <path d="M30 6 C30 6 14 26 14 36 C14 45 21 52 30 52 C39 52 46 45 46 36 C46 26 30 6 30 6 Z"
                fill="url(#sdropG)" filter="url(#sglow)"/>
          <ellipse cx="23" cy="28" rx="3.5" ry="6" fill="white" opacity="0.3" transform="rotate(-15 23 28)"/>
          <path d="M22 36 L28 43 L40 28" stroke="white" stroke-width="3"
                stroke-linecap="round" stroke-linejoin="round" fill="none"/>
          <text x="56" y="34" font-family="Inter,Arial,sans-serif" font-size="22"
                font-weight="900" fill="#7dd3fc" letter-spacing="-0.5">Aqua</text>
          <text x="103" y="34" font-family="Inter,Arial,sans-serif" font-size="22"
                font-weight="900" fill="#ffffff" letter-spacing="-0.5">Check</text>
          <text x="57" y="46" font-family="Inter,Arial,sans-serif" font-size="7.5"
                font-weight="600" fill="#7dd3fc" letter-spacing="1.8">WATER SAFETY AI</text>
        </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 2. Navigation links ──
    st.sidebar.page_link("app.py",             label="🏠  Home")
    st.sidebar.page_link("pages/1_Predict.py", label="🔬  Check My Water")
    st.sidebar.page_link("pages/2_Insights.py",label="🧠  How It Works")
    st.sidebar.page_link("pages/3_About.py",   label="📖  About")

    # ── 3. Accuracy + fact at bottom ──
    st.sidebar.markdown(
        f"""
        <div style="margin-top:40px;border-top:1px solid rgba(255,255,255,0.10);padding-top:16px;">
            <div class="sidebar-acc-box">
                <div class="sidebar-acc-label">Model Accuracy</div>
                <div class="sidebar-acc-value">{acc:.0f}%</div>
                <div class="sidebar-acc-sub">on real water samples</div>
            </div>
            <div class="fact-box">
                <strong style="color:#38bdf8 !important;">Did you know?</strong><br>{fact}
            </div>
            <div style="text-align:center;margin-top:14px;font-size:0.72rem;color:#4a8fa8 !important;">
                &#169; AquaCheck &nbsp;&middot;&nbsp; Water Safety AI
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero(eyebrow: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <section class="app-hero">
            <div style="margin-bottom:14px;">
            <svg viewBox="0 0 200 60" xmlns="http://www.w3.org/2000/svg" style="height:44px;">
              <defs>
                <linearGradient id="hdropG" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stop-color="#ffffff"/>
                  <stop offset="100%" stop-color="#bae6fd"/>
                </linearGradient>
              </defs>
              <path d="M30 6 C30 6 14 26 14 36 C14 45 21 52 30 52 C39 52 46 45 46 36 C46 26 30 6 30 6 Z"
                    fill="url(#hdropG)" opacity="0.95"/>
              <ellipse cx="23" cy="28" rx="3.5" ry="6" fill="white" opacity="0.4" transform="rotate(-15 23 28)"/>
              <path d="M22 36 L28 43 L40 28" stroke="#0369a1" stroke-width="3"
                    stroke-linecap="round" stroke-linejoin="round" fill="none"/>
              <text x="56" y="34" font-family="Inter,Arial,sans-serif" font-size="22"
                    font-weight="900" fill="#ffffff" letter-spacing="-0.5">Aqua</text>
              <text x="103" y="34" font-family="Inter,Arial,sans-serif" font-size="22"
                    font-weight="900" fill="#bae6fd" letter-spacing="-0.5">Check</text>
              <text x="57" y="46" font-family="Inter,Arial,sans-serif" font-size="7.5"
                    font-weight="600" fill="#e0f2fe" letter-spacing="1.8" opacity="0.8">WATER SAFETY AI</text>
            </svg>
            </div>
            <div class="eyebrow">{eyebrow}</div>
            <div class="hero-title">{title}</div>
            <p class="hero-copy">{copy}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def step_card(icon: str, step: str, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="step-card">
            <div class="step-icon">{icon}</div>
            <div class="step-num">STEP {step}</div>
            <div class="step-title">{title}</div>
            <div class="step-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_card(icon: str, value: str, label: str) -> None:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-icon">{icon}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(icon: str, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-card-icon">{icon}</div>
            <div class="info-card-title">{title}</div>
            <div class="info-card-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def confidence_bar(probability: float, is_safe: bool) -> None:
    pct = int(probability * 100)
    cls = "conf-safe" if is_safe else "conf-risk"
    st.markdown(
        f"""
        <div style="margin:16px 0 8px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                <span style="font-size:0.85rem;font-weight:700;color:#1e3a52;">Confidence Level</span>
                <span style="font-size:0.85rem;font-weight:800;color:{'#059669' if is_safe else '#dc2626'};">{pct}%</span>
            </div>
            <div class="conf-bar-wrap">
                <div class="conf-bar-fill {cls}" style="width:{pct}%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def feature_importance_bars(importance_df) -> None:
    max_imp = importance_df["Importance"].max()
    for _, row in importance_df.iterrows():
        pct = (row["Importance"] / max_imp) * 100
        st.markdown(
            f"""
            <div class="feat-row">
                <div class="feat-name">{row['What it measures']}</div>
                <div class="feat-bar-bg">
                    <div class="feat-bar-fill" style="width:{pct:.1f}%;"></div>
                </div>
                <div class="feat-pct">{row['Importance']:.3f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def disclaimer() -> None:
    st.markdown(
        """
        <div class="disclaimer">
            ⚠️ <strong>Educational use only.</strong> This app provides an AI-based estimate.
            Always consult a certified water testing laboratory for official safety decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )


def footer() -> None:
    st.markdown(
        """
        <div class="footer">
            💧 <strong>AquaCheck</strong> &nbsp;·&nbsp; AI-powered water potability prediction
            &nbsp;·&nbsp; Built with XGBoost &amp; Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )
