"""
pages/02_Classification_History.py — EcoSort AI
==================================================
Classification History page.

Shows all past classification records stored in the local SQLite database.

Layout (when records exist):
  ┌─────────────────────────────────────────────────────┐
  │  Summary metrics: total, most common, avg confidence │
  ├──────────────────┬──────────────────────────────────┤
  │  Donut pie chart │  Category bar chart               │
  ├──────────────────┴──────────────────────────────────┤
  │  Daily trend line chart                             │
  ├──────────────────────────────────────────────────────┤
  │  Full history table (pandas DataFrame)              │
  ├──────────────────────────────────────────────────────┤
  │  ⚠️ Clear history button (requires confirmation)    │
  └──────────────────────────────────────────────────────┘

Empty state:
  A friendly notice is shown when there are no records. No sample or fake
  data is generated.

Clear history:
  A two-step confirmation flow prevents accidental deletion:
    Step 1 — user clicks "Clear All History"
    Step 2 — a warning is shown; user must click "Yes, delete everything"
  Cancelling returns to the normal view without deleting anything.
"""

import pandas as pd
import streamlit as st
from collections import Counter

from data.database import init_db, fetch_history, clear_history, record_count
from components.ui_helpers import (
    render_page_header,
    render_notice,
    render_warning,
    render_divider,
    render_metric_row,
)
from components.charts import (
    history_category_pie,
    history_category_bar,
    history_trend_line,
)

# ── Database initialisation ───────────────────────────────────────────────────
init_db()

# ── Page header ───────────────────────────────────────────────────────────────
render_page_header(
    "📊 Classification History",
    "Review all past waste classifications saved during this session.",
)

# ── Fetch records ─────────────────────────────────────────────────────────────
records = fetch_history(limit=500)

if not records:
    render_notice(
        "No classification history yet.<br><br>"
        "Go to <strong>Classify Waste</strong> and upload an image to get started. "
        "Each successful classification is automatically saved here.",
        icon="📭",
    )
    st.stop()

# ── Build DataFrame ───────────────────────────────────────────────────────────
df = pd.DataFrame(records)

# Parse ISO timestamp to a readable local string and extract the date part
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df["date"]      = df["timestamp"].dt.strftime("%Y-%m-%d")
df["time"]      = df["timestamp"].dt.strftime("%H:%M UTC")
df["confidence_pct"] = (df["confidence"] * 100).round(1).astype(str) + "%"

# ── Summary metrics ───────────────────────────────────────────────────────────
total        = len(df)
most_common  = df["predicted_label"].mode().iloc[0].capitalize() if total else "—"
avg_conf_pct = f"{df['confidence'].mean() * 100:.1f}%"

render_metric_row([
    {"icon": "📊", "label": "Total Classifications", "value": str(total)},
    {"icon": "♻️", "label": "Most Common Category",  "value": most_common},
    {"icon": "🎯", "label": "Average Confidence",    "value": avg_conf_pct},
])

render_divider()

# ── Category charts ───────────────────────────────────────────────────────────
category_counts = dict(Counter(df["predicted_label"].str.lower()))

col_pie, col_bar = st.columns(2, gap="large")

with col_pie:
    fig_pie = history_category_pie(category_counts)
    st.plotly_chart(fig_pie, use_container_width=True)

with col_bar:
    fig_bar = history_category_bar(category_counts)
    st.plotly_chart(fig_bar, use_container_width=True)

render_divider()

# ── Daily trend line ──────────────────────────────────────────────────────────
# Count classifications per date and sort chronologically
daily_series = df.groupby("date").size().sort_index()
if len(daily_series) >= 2:
    # Only show the trend chart when there is more than one day of data
    daily_counts = daily_series.to_dict()
    fig_trend = history_trend_line(daily_counts)
    st.plotly_chart(fig_trend, use_container_width=True)
    render_divider()

# ── History table ─────────────────────────────────────────────────────────────
st.markdown(
    "<div style='font-size:0.78rem; color:#57606a; margin-bottom:0.5rem;"
    "text-transform:uppercase; letter-spacing:0.06em;'>"
    "All Records</div>",
    unsafe_allow_html=True,
)

# Build a clean display DataFrame (drop internal columns)
display_df = df[["id", "date", "time", "filename", "predicted_label", "confidence_pct"]].copy()
display_df.columns = ["ID", "Date", "Time (UTC)", "Filename", "Category", "Confidence"]
display_df["Category"] = display_df["Category"].str.capitalize()

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
)

render_divider()

# ── Clear history — two-step confirmation ─────────────────────────────────────
# Session state key tracks whether the user has clicked the first button.
if "confirm_clear" not in st.session_state:
    st.session_state.confirm_clear = False

if not st.session_state.confirm_clear:
    if st.button("🗑️ Clear All History", type="secondary"):
        st.session_state.confirm_clear = True
        st.rerun()
else:
    render_warning(
        "<strong>This will permanently delete all classification records.</strong><br>"
        "This action cannot be undone.",
        icon="⚠️",
    )
    st.markdown("<br>", unsafe_allow_html=True)

    col_yes, col_no, _ = st.columns([1, 1, 4])
    with col_yes:
        if st.button("✅ Yes, delete everything", type="primary"):
            clear_history()
            st.session_state.confirm_clear = False
            st.success("History cleared.")
            st.rerun()
    with col_no:
        if st.button("❌ Cancel"):
            st.session_state.confirm_clear = False
            st.rerun()
