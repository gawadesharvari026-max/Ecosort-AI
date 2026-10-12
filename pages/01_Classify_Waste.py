"""
pages/01_Classify_Waste.py — EcoSort AI
==========================================
Classify Waste page.

User flow:
  1. User uploads a JPG / JPEG / PNG image.
  2. The image is displayed.
  3. If the model weights file does not exist → show the model-not-configured
     notice. No prediction is attempted. No fake result is shown.
  4. If the model IS available → run inference → display:
       • Predicted category and confidence badge
       • Bin-stream badge and accepted-by line
       • Horizontal confidence bar chart (all 6 classes)
       • Recycling and disposal recommendation card
  5. The result is saved to SQLite via data/database.insert_record().
  6. Errors from image loading or model inference are caught and shown
     as user-friendly warning banners, never as raw tracebacks.

Model caching
-------------
st.cache_resource is used to load the model once per Streamlit server
session. The model is NOT reloaded on every page interaction — only when
the server restarts or the cache is cleared. This keeps the UI responsive.
"""

import streamlit as st
from PIL import Image, UnidentifiedImageError

from model.classifier import (
    classify_image,
    load_model,
    is_model_available,
    ModelNotConfiguredError,
)
from data.database import init_db, insert_record
from data.recommendations import get_recommendation
from components.ui_helpers import (
    render_page_header,
    render_model_not_configured_notice,
    render_result_card,
    render_recommendation_card,
    render_warning,
    render_notice,
    render_divider,
)
from components.charts import confidence_bar_chart

# ── Database initialisation ───────────────────────────────────────────────────
# Safe to call on every page load — no-op if the table already exists.
init_db()


# ── Cached model loader ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model weights…")
def _get_model():
    """
    Load and cache the MobileNetV2 weights for the session.

    st.cache_resource keeps the returned object alive across reruns so the
    model file is read from disk only once. If the weights file is missing,
    load_model() raises ModelNotConfiguredError which is caught below.
    """
    return load_model()


# ── Page header ───────────────────────────────────────────────────────────────
render_page_header(
    "🔍 Classify Waste",
    "Upload a photo of a waste item to identify its category and get recycling advice.",
)

# ── Model status check ────────────────────────────────────────────────────────
if not is_model_available():
    render_model_not_configured_notice()
    st.stop()   # Nothing more to render without a model

# ── Image uploader ────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Choose a waste image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG. Maximum file size: 200 MB.",
)

if uploaded_file is None:
    render_notice(
        "Upload an image above to get started. The model will identify the waste "
        "category and provide recycling and disposal recommendations.",
        icon="📷",
    )
    st.stop()

# ── Image display ─────────────────────────────────────────────────────────────
# Show the uploaded image before running inference so the user can see what
# was uploaded even if inference fails.
col_img, col_result = st.columns([1, 2], gap="large")

with col_img:
    try:
        image = Image.open(uploaded_file)
    except UnidentifiedImageError:
        render_warning(
            "The uploaded file could not be opened as an image. "
            "Please upload a valid JPG or PNG file."
        )
        st.stop()
    except Exception as exc:
        render_warning(f"Unexpected error reading the image file: {exc}")
        st.stop()

    st.image(image, caption=uploaded_file.name, use_container_width=True)

# ── Inference ─────────────────────────────────────────────────────────────────
with col_result:
    with st.spinner("Classifying…"):
        try:
            model  = _get_model()
            result = classify_image(image, model=model)
        except ModelNotConfiguredError:
            # Edge case: file was deleted between is_model_available() and here
            render_model_not_configured_notice()
            st.stop()
        except Exception as exc:
            render_warning(
                f"<strong>Classification failed.</strong><br>"
                f"The model encountered an error while processing this image.<br>"
                f"<small>Details: {exc}</small>"
            )
            st.stop()

    # ── Fetch matching recommendation ─────────────────────────────────────────
    recommendation = get_recommendation(result.label)

    # ── Save to database ──────────────────────────────────────────────────────
    try:
        insert_record(
            filename        = uploaded_file.name,
            predicted_label = result.label,
            confidence      = result.confidence,
            recommendation  = recommendation.get("tip", ""),
            username        = st.session_state.get("username", ""),
        )
    except Exception as exc:
        # Database errors should not prevent the result from being shown
        render_warning(
            f"Result displayed but could not be saved to history. "
            f"<small>({exc})</small>"
        )

    # ── Result card ───────────────────────────────────────────────────────────
    render_result_card(result, recommendation)

    # ── Confidence bar chart ──────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:0.78rem; color:#57606a; margin-bottom:0.4rem;"
        "text-transform:uppercase; letter-spacing:0.06em;'>"
        "Model confidence — all categories</div>",
        unsafe_allow_html=True,
    )
    fig = confidence_bar_chart(result.all_scores)
    st.plotly_chart(fig, use_container_width=True)

# ── Recommendation card (full width below the two columns) ───────────────────
render_divider()
render_recommendation_card(recommendation)

render_divider()
st.caption(
    "⚠️ AI predictions can be incorrect. Always verify disposal guidance "
    "with your local council or waste authority."
)
