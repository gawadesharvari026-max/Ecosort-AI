"""
app.py — EcoSort AI
====================
Main Streamlit entry point.

Responsibilities:
- Sets the global page configuration (title, icon, layout).
- Injects global CSS that complements the .streamlit/config.toml theme.
- Renders the sidebar header and branding.
- Shows a friendly notice when the ML model has not yet been configured.

Streamlit automatically discovers pages in the /pages folder and adds
them to the sidebar navigation. This file is the shell that wraps them.
"""

import streamlit as st

# ── Page configuration ────────────────────────────────────────────────────────
# Must be the very first Streamlit call in the script.
st.set_page_config(
    page_title="EcoSort AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
# Injects styles that config.toml cannot express:
# custom card surfaces, mint accents, badge colours, and font tweaks.
st.markdown(
    """
    <style>
    /* ── Typography ───────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
    }

    /* ── Sidebar branding strip ───────────────────────────────────── */
    [data-testid="stSidebar"] {
        border-right: 2px solid #95D5B2;
    }
    [data-testid="stSidebar"] .stMarkdown h1 {
        color: #2D6A4F;
        font-size: 1.3rem;
        margin-bottom: 0.1rem;
    }

    /* ── White card surface ───────────────────────────────────────── */
    .eco-card {
        background: #ffffff;
        border: 1px solid #D8EAD3;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }

    /* ── Mint highlight tag ───────────────────────────────────────── */
    .eco-badge {
        display: inline-block;
        background: #95D5B2;
        color: #1B2F1E;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        letter-spacing: 0.03em;
    }

    /* ── Info / notice banner ─────────────────────────────────────── */
    .eco-notice {
        background: #EEF7F1;
        border-left: 4px solid #2D6A4F;
        border-radius: 0 8px 8px 0;
        padding: 0.9rem 1.2rem;
        color: #1B2F1E;
        font-size: 0.93rem;
        line-height: 1.6;
    }

    /* ── Warning banner ───────────────────────────────────────────── */
    .eco-warning {
        background: #FFF8E7;
        border-left: 4px solid #D4A017;
        border-radius: 0 8px 8px 0;
        padding: 0.9rem 1.2rem;
        color: #4A3500;
        font-size: 0.93rem;
        line-height: 1.6;
    }

    /* ── Divider ──────────────────────────────────────────────────── */
    .eco-divider {
        border: none;
        border-top: 1px solid #D8EAD3;
        margin: 1.2rem 0;
    }

    /* ── Hide default Streamlit footer ───────────────────────────── */
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# ♻️ EcoSort AI")
    st.markdown(
        "<span class='eco-badge'>Waste Classification System</span>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='eco-divider'>", unsafe_allow_html=True)
    st.caption(
        "Upload a waste image to identify its category and receive "
        "recycling and disposal recommendations."
    )
    st.markdown("<hr class='eco-divider'>", unsafe_allow_html=True)
    st.caption("Navigate using the pages above ↑")

# ── Home page content ─────────────────────────────────────────────────────────
st.markdown("## Welcome to EcoSort AI ♻️")
st.markdown(
    "An intelligent waste classification and recycling recommendation system."
)

st.markdown("<hr class='eco-divider'>", unsafe_allow_html=True)

# Model status notice — honest about the current state
st.markdown(
    """
    <div class='eco-notice'>
        <strong>ℹ️ Model Status: Not Configured</strong><br>
        No trained waste classification model has been loaded yet.
        The <em>Classify Waste</em> page will show this same notice until a
        compatible model file is placed in the <code>model/</code> folder and
        the category labels are confirmed.<br><br>
        This message will be removed automatically once the model is set up.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ── Feature overview cards ────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class='eco-card'>
            <strong>🔍 Classify Waste</strong>
            <hr class='eco-divider'>
            Upload a photo of any waste item. The AI model will identify
            its category and confidence level.
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class='eco-card'>
            <strong>♻️ Recycling Advice</strong>
            <hr class='eco-divider'>
            Each classification comes with clear recycling and disposal
            recommendations — separate from the ML prediction.
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class='eco-card'>
            <strong>📊 Classification History</strong>
            <hr class='eco-divider'>
            Every classification is saved locally using SQLite so you can
            review trends and past results at any time.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr class='eco-divider'>", unsafe_allow_html=True)
st.caption(
    "EcoSort AI — Machine Learning & Applied AI Internship Project · "
    "Built with Streamlit and PyTorch"
)
