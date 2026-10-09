"""
components/charts.py — EcoSort AI
=====================================
Plotly chart builders for the EcoSort AI interface.

Every function returns a plotly.graph_objects.Figure that the calling page
renders with st.plotly_chart(fig, use_container_width=True).

Charts never generate or contain hardcoded data. All functions require real
data to be passed in — if there is no data, the page is responsible for
showing an appropriate empty-state message instead of calling these functions.

Design palette (matches config.toml and app.py):
  Forest green  #2D6A4F
  Mint          #95D5B2
  Off-white     #F8FAF5  (chart paper/plot background)
  White         #ffffff  (chart area background)
  Deep green    #1B2F1E  (axis text)
  Muted         #57606a  (gridlines, subtitles)

Public API
----------
  confidence_bar_chart(all_scores)
      Horizontal bar chart of per-class softmax scores from one inference.

  history_category_pie(category_counts)
      Pie chart of total classifications by waste category.

  history_category_bar(category_counts)
      Vertical bar chart of total classifications by waste category.

  history_trend_line(daily_counts)
      Line chart of daily classification volume over time.
"""

import plotly.graph_objects as go
import plotly.express as px

# ── Shared theme values ───────────────────────────────────────────────────────
_GREEN      = "#2D6A4F"
_MINT       = "#95D5B2"
_OFF_WHITE  = "#F8FAF5"
_TEXT       = "#1B2F1E"
_MUTED      = "#57606a"
_PAPER      = "#F8FAF5"   # chart paper (outer) background
_PLOT_BG    = "#ffffff"   # chart plot (inner) background
_GRIDLINE   = "#E5EDE2"   # subtle green-tinted gridlines

# Colour sequence used for multi-category charts (6 TrashNet classes)
# Ordered to match INDEX_TO_LABEL: cardboard, glass, metal, paper, plastic, trash
_CATEGORY_COLOURS: dict[str, str] = {
    "cardboard": "#A0522D",   # brown
    "glass":     "#2E8B57",   # sea green
    "metal":     "#708090",   # slate
    "paper":     "#4682B4",   # steel blue
    "plastic":   "#FF8C00",   # dark orange
    "trash":     "#696969",   # dim grey
}

# Fallback palette for any unlisted category
_FALLBACK_COLOURS = [
    "#2D6A4F", "#95D5B2", "#4682B4", "#A0522D", "#708090", "#FF8C00",
]


def _base_layout(title: str = "", height: int = 360) -> dict:
    """Return a shared Plotly layout dict applying the EcoSort theme."""
    return dict(
        title=dict(
            text=title,
            font=dict(size=14, color=_TEXT, family="Segoe UI, sans-serif"),
            x=0,
            xanchor="left",
            pad=dict(l=0, t=4),
        ),
        paper_bgcolor=_PAPER,
        plot_bgcolor=_PLOT_BG,
        font=dict(family="Segoe UI, system-ui, sans-serif", color=_TEXT, size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        height=height,
        showlegend=False,
    )


# ── Confidence bar chart ──────────────────────────────────────────────────────

def confidence_bar_chart(all_scores: dict[str, float]) -> go.Figure:
    """
    Horizontal bar chart showing the model's softmax probability for every
    TrashNet class after a single inference.

    The top-predicted bar is highlighted in forest green; all others are mint.
    Values are displayed as percentages on the x-axis (0 – 100%).

    Parameters
    ----------
    all_scores : Dict mapping label string → softmax probability (0.0–1.0).
                 Comes directly from ClassificationResult.all_scores.
                 Must contain at least one entry.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if not all_scores:
        raise ValueError("all_scores must not be empty.")

    # Sort bars by probability descending so the top prediction is at the top
    sorted_items = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0].capitalize() for item in sorted_items]
    values = [item[1] * 100 for item in sorted_items]   # convert to %

    # Highlight the top bar; all others get mint
    bar_colours = [_GREEN if i == 0 else _MINT for i in range(len(labels))]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=bar_colours,
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            textfont=dict(size=11, color=_TEXT),
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )

    layout = _base_layout("Confidence Scores — All Categories", height=300)
    layout.update(
        xaxis=dict(
            range=[0, 105],
            ticksuffix="%",
            showgrid=True,
            gridcolor=_GRIDLINE,
            zeroline=False,
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=12),
        ),
    )
    fig.update_layout(**layout)
    return fig


# ── History — category pie chart ──────────────────────────────────────────────

def history_category_pie(category_counts: dict[str, int]) -> go.Figure:
    """
    Pie chart showing the proportion of each waste category in the
    classification history.

    Parameters
    ----------
    category_counts : Dict mapping label string → count of classifications.
                      Must have at least one entry with a count > 0.
                      Example: {"plastic": 12, "glass": 5, "metal": 3}

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if not category_counts:
        raise ValueError("category_counts must not be empty.")

    labels = [k.capitalize() for k in category_counts.keys()]
    values = list(category_counts.values())
    colours = [
        _CATEGORY_COLOURS.get(k.lower(), _FALLBACK_COLOURS[i % len(_FALLBACK_COLOURS)])
        for i, k in enumerate(category_counts.keys())
    ]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            marker=dict(
                colors=colours,
                line=dict(color=_PAPER, width=2),
            ),
            textinfo="label+percent",
            textfont=dict(size=12),
            hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
            hole=0.35,   # donut style — more modern than a full pie
        )
    )

    layout = _base_layout("Classifications by Category", height=360)
    layout["showlegend"] = True
    layout["legend"] = dict(
        orientation="v",
        x=1.02,
        y=0.5,
        font=dict(size=11),
    )
    fig.update_layout(**layout)
    return fig


# ── History — category bar chart ─────────────────────────────────────────────

def history_category_bar(category_counts: dict[str, int]) -> go.Figure:
    """
    Vertical bar chart showing total classifications per waste category.

    Parameters
    ----------
    category_counts : Dict mapping label string → count.
                      Example: {"plastic": 12, "glass": 5, "metal": 3}

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if not category_counts:
        raise ValueError("category_counts must not be empty.")

    # Sort by count descending
    sorted_items = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    labels  = [k.capitalize() for k, _ in sorted_items]
    counts  = [v for _, v in sorted_items]
    colours = [
        _CATEGORY_COLOURS.get(k.lower(), _GREEN)
        for k, _ in sorted_items
    ]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=counts,
            marker_color=colours,
            text=counts,
            textposition="outside",
            textfont=dict(size=12, color=_TEXT),
            hovertemplate="%{x}: %{y} classifications<extra></extra>",
        )
    )

    layout = _base_layout("Total Classifications by Category", height=340)
    layout.update(
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=_GRIDLINE,
            zeroline=False,
            tickfont=dict(size=11),
            title=dict(text="Count", font=dict(size=12, color=_MUTED)),
        ),
        bargap=0.35,
    )
    fig.update_layout(**layout)
    return fig


# ── History — daily trend line ────────────────────────────────────────────────

def history_trend_line(daily_counts: dict[str, int]) -> go.Figure:
    """
    Line chart showing the number of classifications performed each day.

    Parameters
    ----------
    daily_counts : Dict mapping date string (YYYY-MM-DD) → count.
                   Must be sorted chronologically (oldest first).
                   Example: {"2025-06-01": 3, "2025-06-02": 7, "2025-06-03": 2}

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if not daily_counts:
        raise ValueError("daily_counts must not be empty.")

    dates  = list(daily_counts.keys())
    counts = list(daily_counts.values())

    fig = go.Figure()

    # Filled area under the line for visual weight
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=counts,
            mode="lines+markers",
            line=dict(color=_GREEN, width=2.5),
            marker=dict(color=_GREEN, size=7),
            fill="tozeroy",
            fillcolor="rgba(45,106,79,0.10)",   # forest green at 10% opacity
            hovertemplate="%{x}: %{y} classifications<extra></extra>",
        )
    )

    layout = _base_layout("Daily Classification Volume", height=300)
    layout.update(
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=11),
            tickangle=-30,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=_GRIDLINE,
            zeroline=False,
            tickfont=dict(size=11),
            title=dict(text="Classifications", font=dict(size=12, color=_MUTED)),
            rangemode="tozero",
        ),
    )
    fig.update_layout(**layout)
    return fig
