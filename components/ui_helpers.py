"""
components/ui_helpers.py — EcoSort AI
========================================
Reusable Streamlit UI components for the EcoSort AI interface.

Every function in this module renders HTML via st.markdown(unsafe_allow_html=True).
All CSS classes referenced here (.eco-card, .eco-badge, .eco-notice, .eco-warning,
.eco-divider) are defined once in app.py's global <style> block and are available
on every page because Streamlit's CSS injection persists across page navigations.

Design palette (matches config.toml and app.py):
  Forest green  #2D6A4F   — primary accent, headings
  Mint          #95D5B2   — secondary highlights, badges
  Off-white     #F8FAF5   — page background
  White         #ffffff   — card surfaces
  Deep green    #1B2F1E   — body text

Public API
----------
  render_page_header(title, subtitle)
  render_divider()
  render_notice(message, icon)
  render_warning(message, icon)
  render_model_not_configured_notice()
  render_eco_badge(text, colour)
  render_confidence_badge(confidence)
  render_bin_badge(bin_label, bin_colour)
  render_result_card(result, recommendation)
  render_metric_row(metrics)
  render_recommendation_card(recommendation)
"""

import streamlit as st


# ── Palette constants (single source of truth for inline styles) ─────────────
_GREEN      = "#2D6A4F"
_MINT       = "#95D5B2"
_OFF_WHITE  = "#F8FAF5"
_CARD_BG    = "#ffffff"
_TEXT       = "#1B2F1E"
_BORDER     = "#D8EAD3"
_MUTED      = "#57606a"


# ── Layout helpers ────────────────────────────────────────────────────────────

def render_page_header(title: str, subtitle: str = "") -> None:
    """
    Render a styled page title and optional subtitle.

    Parameters
    ----------
    title    : Main heading text (e.g. "♻️ Classify Waste").
    subtitle : Optional one-line description shown below the title.
    """
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(
            f"<p style='color:{_MUTED}; margin-top:-0.5rem; font-size:0.95rem;'>"
            f"{subtitle}</p>",
            unsafe_allow_html=True,
        )
    render_divider()


def render_divider() -> None:
    """Render a thin green-tinted horizontal rule."""
    st.markdown("<hr class='eco-divider'>", unsafe_allow_html=True)


# ── Notice / warning banners ──────────────────────────────────────────────────

def render_notice(message: str, icon: str = "ℹ️") -> None:
    """
    Render a forest-green left-bordered info notice.

    Parameters
    ----------
    message : HTML-safe message string (may include <br> and <strong> tags).
    icon    : Emoji or text icon shown before the message.
    """
    st.markdown(
        f"<div class='eco-notice'>{icon} {message}</div>",
        unsafe_allow_html=True,
    )


def render_warning(message: str, icon: str = "⚠️") -> None:
    """
    Render an amber left-bordered warning notice.

    Parameters
    ----------
    message : HTML-safe message string.
    icon    : Emoji or text icon shown before the message.
    """
    st.markdown(
        f"<div class='eco-warning'>{icon} {message}</div>",
        unsafe_allow_html=True,
    )


def render_model_not_configured_notice() -> None:
    """
    Render the standard 'model not yet configured' full notice block.
    Used on any page that requires inference but has no .pth file available.
    No predictions are shown — this is the honest fallback state.
    """
    st.markdown(
        """
        <div class='eco-notice'>
            <strong>ℹ️ Model Not Configured</strong><br><br>
            No trained model weights file was found at
            <code>model/trashnet_mobilenetv2.pth</code>.<br><br>
            To enable waste classification:<br>
            &nbsp;&nbsp;1. Download the <strong>TrashNet</strong> dataset
               (<a href="https://github.com/garythung/trashnet"
                  target="_blank">github.com/garythung/trashnet</a>).<br>
            &nbsp;&nbsp;2. Run the training script to produce
               <code>trashnet_mobilenetv2.pth</code>.<br>
            &nbsp;&nbsp;3. Place the <code>.pth</code> file in the
               <code>model/</code> folder.<br>
            &nbsp;&nbsp;4. Restart the app.<br><br>
            No predictions will be shown until this file is present.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Badges ────────────────────────────────────────────────────────────────────

def render_eco_badge(text: str, colour: str = _MINT) -> None:
    """
    Render a small pill-shaped badge inline.

    Parameters
    ----------
    text   : Badge label text.
    colour : Background hex colour (default: mint #95D5B2).
    """
    # Choose black or white text based on a simple luminance estimate
    text_colour = _TEXT if colour in (_MINT, _OFF_WHITE, "#EEF7F1") else "#ffffff"
    st.markdown(
        f"<span class='eco-badge' style='background:{colour}; color:{text_colour};'>"
        f"{text}</span>",
        unsafe_allow_html=True,
    )


def render_confidence_badge(confidence: float) -> None:
    """
    Render a colour-coded confidence level badge.

    Colour scale:
      ≥ 0.80  → forest green  (high confidence)
      ≥ 0.50  → amber         (medium confidence)
      < 0.50  → red-brown     (low confidence — treat result with caution)

    Parameters
    ----------
    confidence : Float between 0.0 and 1.0 from the model's softmax output.
    """
    pct = confidence * 100
    if confidence >= 0.80:
        colour, label = _GREEN, "High confidence"
    elif confidence >= 0.50:
        colour, label = "#D4A017", "Medium confidence"
    else:
        colour, label = "#B94040", "Low confidence — treat with caution"

    st.markdown(
        f"<span style='"
        f"display:inline-block; background:{colour}; color:#ffffff;"
        f"font-size:0.8rem; font-weight:600; padding:0.25rem 0.8rem;"
        f"border-radius:20px; letter-spacing:0.02em;'>"
        f"{pct:.1f}% — {label}"
        f"</span>",
        unsafe_allow_html=True,
    )


def render_bin_badge(bin_label: str, bin_colour: str) -> None:
    """
    Render a coloured bin-stream badge (e.g. "♻️ Recycling" in brown).

    Parameters
    ----------
    bin_label  : Short description (e.g. "Glass Recycling").
    bin_colour : Hex colour for the badge background.
    """
    st.markdown(
        f"<span style='"
        f"display:inline-block; background:{bin_colour}; color:#ffffff;"
        f"font-size:0.82rem; font-weight:600; padding:0.25rem 0.9rem;"
        f"border-radius:20px;'>"
        f"🗑️ {bin_label}"
        f"</span>",
        unsafe_allow_html=True,
    )


# ── Classification result card ────────────────────────────────────────────────

def render_result_card(result, recommendation: dict) -> None:
    """
    Render the main classification result card shown after a successful inference.

    Displays:
      - Predicted waste category (large, green)
      - Confidence badge
      - Bin-stream badge
      - Accepted-by sub-heading
      - Separator before the recommendation section

    Parameters
    ----------
    result         : A ClassificationResult dataclass from model/classifier.py.
                     Expected attributes: display_name (str), confidence (float).
    recommendation : A dict from data/recommendations.get_recommendation().
                     Expected keys: bin_label, bin_colour, accepted_by.
    """
    display_name = result.display_name
    confidence   = result.confidence
    bin_label    = recommendation.get("bin_label", "Unknown")
    bin_colour   = recommendation.get("bin_colour", "#888888")
    accepted_by  = recommendation.get("accepted_by", "")

    st.markdown(
        f"""
        <div class='eco-card'>
            <div style='font-size:0.78rem; color:{_MUTED}; margin-bottom:0.3rem;
                        text-transform:uppercase; letter-spacing:0.06em;'>
                Predicted Category
            </div>
            <div style='font-size:2rem; font-weight:700; color:{_GREEN};
                        line-height:1.2; margin-bottom:0.6rem;'>
                {display_name}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Confidence and bin badges on the same row
    col_conf, col_bin, _ = st.columns([2, 2, 3])
    with col_conf:
        render_confidence_badge(confidence)
    with col_bin:
        render_bin_badge(bin_label, bin_colour)

    if accepted_by:
        st.markdown(
            f"<p style='color:{_MUTED}; font-size:0.87rem; margin-top:0.6rem;'>"
            f"📦 {accepted_by}</p>",
            unsafe_allow_html=True,
        )

    render_divider()


# ── Recommendation card ───────────────────────────────────────────────────────

def render_recommendation_card(recommendation: dict) -> None:
    """
    Render the recycling and disposal recommendation card.

    Displays:
      - Numbered list of disposal steps
      - Contextual tip in a highlighted notice block

    This content comes from data/recommendations.py — it is static,
    human-written guidance completely separate from the ML prediction.

    Parameters
    ----------
    recommendation : Dict from data/recommendations.get_recommendation().
                     Expected keys: steps (list[str]), tip (str).
    """
    steps = recommendation.get("steps", [])
    tip   = recommendation.get("tip", "")

    st.markdown(
        f"<div style='font-size:0.78rem; color:{_MUTED}; margin-bottom:0.5rem;"
        f"text-transform:uppercase; letter-spacing:0.06em;'>"
        f"♻️ Recycling & Disposal Guide</div>",
        unsafe_allow_html=True,
    )

    if steps:
        # Build a styled numbered list inside a white card
        items_html = "".join(
            f"<li style='margin-bottom:0.45rem; line-height:1.55;'>{step}</li>"
            for step in steps
        )
        st.markdown(
            f"<div class='eco-card'>"
            f"<ol style='margin:0; padding-left:1.3rem; color:{_TEXT};'>"
            f"{items_html}"
            f"</ol></div>",
            unsafe_allow_html=True,
        )

    if tip:
        st.markdown(
            f"<div class='eco-notice'>"
            f"<strong>💡 Did you know?</strong><br>{tip}"
            f"</div>",
            unsafe_allow_html=True,
        )


# ── Metric row ────────────────────────────────────────────────────────────────

def render_metric_row(metrics: list[dict]) -> None:
    """
    Render a row of summary metric tiles.

    Each metric is displayed in an equal-width column as a white card with
    a large value and a muted label beneath it.

    Parameters
    ----------
    metrics : List of dicts, each with keys:
                label (str)  — e.g. "Total Classifications"
                value (str)  — e.g. "42"
                icon  (str)  — optional emoji, e.g. "📊"

    Example
    -------
    render_metric_row([
        {"icon": "📊", "label": "Total Classifications", "value": "42"},
        {"icon": "♻️", "label": "Most Common Category",  "value": "Plastic"},
    ])
    """
    if not metrics:
        return

    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        icon  = m.get("icon", "")
        label = m.get("label", "")
        value = m.get("value", "—")
        with col:
            st.markdown(
                f"""
                <div class='eco-card' style='text-align:center;'>
                    <div style='font-size:1.7rem; font-weight:700;
                                color:{_GREEN}; line-height:1.1;'>
                        {icon} {value}
                    </div>
                    <div style='font-size:0.8rem; color:{_MUTED};
                                margin-top:0.35rem;'>
                        {label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
