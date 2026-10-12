"""
app.py — EcoSort AI
====================
Main Streamlit entry point.

Responsibilities:
- Sets the global page configuration (title, icon, layout).
- Injects global CSS that complements the .streamlit/config.toml theme.
- Enforces a login gate — any registered user can sign in.
- New users can create an account via the Register tab.
- Renders the sidebar header, branding, and logout button when logged in.
- Shows a friendly notice when the ML model has not yet been configured.

Login system — SQLite user database
-------------------------------------
User accounts are stored in history.db (users table).
Passwords are stored as SHA-256 hashes — never plain text.
Any number of users can register. A default admin/ecosort123 account
is seeded automatically on first run.
"""

import streamlit as st

# ── Page configuration ────────────────────────────────────────────────────────
# Must be the very first Streamlit call in the script.
st.set_page_config(
    page_title="EcoSort AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed",  # collapsed so sidebar CSS hide works cleanly
)

# ── Database — initialise on every startup ───────────────────────────────────
# Creates the users + classifications tables and seeds the default admin
# account if it does not exist yet. Safe to call repeatedly.
from data.database import init_db, register_user, verify_user, user_exists
init_db()

# ── Session state — initialise before any rendering ──────────────────────────
# Must be set before the CSS block so that st.rerun() after login lands on
# the correct branch immediately without a double-render flash.
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""


# ── Global CSS ────────────────────────────────────────────────────────────────
# Injects styles that config.toml cannot express.
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

    /* ── Login card wrapper ───────────────────────────────────────── */
    .login-wrap {
        max-width: 420px;
        margin: 4rem auto 0 auto;
    }

    /* ── Hide default Streamlit footer ───────────────────────────── */
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Login gate ────────────────────────────────────────────────────────────────
# All code below this block is only reached when the user is authenticated.
# st.session_state persists within a single browser session. Closing the tab
# or restarting the Streamlit server clears it automatically.

if not st.session_state.logged_in:
    # Hide the sidebar entirely on the login screen
    st.markdown(
        "<style>[data-testid='stSidebar']{display:none;}</style>",
        unsafe_allow_html=True,
    )

    # Centred card
    st.markdown("<div class='login-wrap'>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style='text-align:center; margin-bottom:1.5rem;'>
            <div style='font-size:3rem;'>♻️</div>
            <div style='font-size:1.6rem; font-weight:700; color:#2D6A4F;'>
                EcoSort AI
            </div>
            <div style='font-size:0.9rem; color:#57606a; margin-top:0.2rem;'>
                Intelligent Waste Classification System
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Login / Register tabs ──────────────────────────────────────────────
    tab_login, tab_register = st.tabs(["🔐 Sign In", "📝 Register"])

    with tab_login:
        with st.form("login_form", clear_on_submit=False):
            st.markdown("##### Welcome back")
            username = st.text_input("Username", placeholder="Enter your username", key="li_user")
            password = st.text_input("Password", type="password", placeholder="Enter your password", key="li_pass")
            submitted = st.form_submit_button("Sign In", use_container_width=True)

            if submitted:
                if not username or not password:
                    st.error("Please enter both username and password.")
                elif verify_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username  = username.strip().lower()
                    st.rerun()
                else:
                    st.error("Incorrect username or password. Please try again.")


    with tab_register:
        with st.form("register_form", clear_on_submit=True):
            st.markdown("##### Create a new account")
            new_user = st.text_input("Choose a username", placeholder="e.g. sharvari", key="reg_user")
            new_pass = st.text_input("Choose a password", type="password", placeholder="At least 6 characters", key="reg_pass")
            confirm  = st.text_input("Confirm password", type="password", placeholder="Re-enter your password", key="reg_conf")
            reg_submitted = st.form_submit_button("Create Account", use_container_width=True)

            if reg_submitted:
                new_user = new_user.strip().lower()
                if not new_user or not new_pass or not confirm:
                    st.error("Please fill in all fields.")
                elif len(new_user) < 3:
                    st.error("Username must be at least 3 characters.")
                elif len(new_pass) < 6:
                    st.error("Password must be at least 6 characters.")
                elif new_pass != confirm:
                    st.error("Passwords do not match.")
                elif user_exists(new_user):
                    st.error(f"Username '{new_user}' is already taken. Please choose another.")
                else:
                    register_user(new_user, new_pass)
                    st.success(f"Account created! Welcome, {new_user.capitalize()} 🎉 Please sign in using the Sign In tab.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Stop here — nothing below is rendered for unauthenticated users
    st.stop()


# ── Authenticated content below this line ────────────────────────────────────

# Expand the sidebar now that the user is logged in
st.markdown(
    "<style>[data-testid='stSidebar']{display:block !important;}</style>",
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

    # Logged-in user display
    username_display = st.session_state.get("username", "user").capitalize()
    st.markdown(
        f"<div style='font-size:0.83rem; color:#2D6A4F; font-weight:600;'>"
        f"👤 {username_display}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # Logout button
    if st.button("🚪 Log Out", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username  = ""
        st.rerun()

    st.markdown("<hr class='eco-divider'>", unsafe_allow_html=True)
    st.caption("Navigate using the pages above ↑")

# ── Home page content ─────────────────────────────────────────────────────────
st.markdown("## Welcome to EcoSort AI ♻️")
st.markdown(
    "An intelligent waste classification and recycling recommendation system."
)

st.markdown("<hr class='eco-divider'>", unsafe_allow_html=True)

from model.classifier import is_model_available

if not is_model_available():
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
else:
    st.markdown(
        """
        <div class='eco-notice'>
            <strong>✅ Model Status: Ready</strong><br>
            The trained waste classification model is loaded and ready.
            Head to <em>Classify Waste</em> to get started.
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
