"""
app.py — EcoSort AI
====================
Main Streamlit entry point.

Responsibilities:
- Sets the global page configuration (title, icon, layout).
- Injects global CSS that complements the .streamlit/config.toml theme.
- Enforces a login gate — all pages are hidden until the user logs in.
- Renders the sidebar header, branding, and logout button when logged in.
- Shows a friendly notice when the ML model has not yet been configured.

Login system (Option A — hardcoded credentials)
------------------------------------------------
Credentials are defined in the USERS dict below. This approach is suitable
for local / internship demos. For a production deployment, move credentials
to .streamlit/secrets.toml or a database with hashed passwords.

Default credentials:
  Username : admin
  Password : ecosort123

To change them, update the USERS dict below.
"""

import hashlib
import streamlit as st

# ── Page configuration ────────────────────────────────────────────────────────
# Must be the very first Streamlit call in the script.
st.set_page_config(
    page_title="EcoSort AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed",  # collapsed so sidebar CSS hide works cleanly
)

# ── Credentials ───────────────────────────────────────────────────────────────
# Format: { "username": "sha256_hex_of_password" }
# Passwords are stored as SHA-256 hashes — never in plain text.
#
# To generate a hash for a new password, run in Python:
#   import hashlib; print(hashlib.sha256("yourpassword".encode()).hexdigest())
#
# Default: admin / ecosort123
USERS: dict[str, str] = {
    "admin": hashlib.sha256("ecosort123".encode()).hexdigest(),
}


def _hash(password: str) -> str:
    """Return the SHA-256 hex digest of a password string."""
    return hashlib.sha256(password.encode()).hexdigest()


def _check_credentials(username: str, password: str) -> bool:
    """Return True if username exists and password matches its stored hash."""
    return USERS.get(username.strip().lower()) == _hash(password)


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

    # Centred login card
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

    with st.form("login_form", clear_on_submit=False):
        st.markdown("#### Sign in to continue")
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        submitted = st.form_submit_button("🔐 Sign In", use_container_width=True)

        if submitted:
            if not username or not password:
                st.error("Please enter both username and password.")
            elif _check_credentials(username, password):
                st.session_state.logged_in = True
                st.session_state.username  = username.strip().lower()
                st.rerun()
            else:
                st.error("Incorrect username or password. Please try again.")

    st.markdown(
        "<p style='text-align:center; color:#57606a; font-size:0.8rem; "
        "margin-top:1rem;'>Default: admin / ecosort123</p>",
        unsafe_allow_html=True,
    )

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
