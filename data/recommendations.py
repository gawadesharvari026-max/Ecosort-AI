"""
data/recommendations.py — EcoSort AI
=======================================
Static recycling and disposal recommendations, keyed by waste category name.

IMPORTANT DESIGN DECISION
--------------------------
This module is entirely independent of the ML model. It is a plain Python
dictionary lookup — no predictions, no probabilities, no AI involved here.

When the classifier returns a predicted label (e.g. "cardboard"), the
calling code uses get_recommendation(label) to fetch the matching advice.
This keeps the ML output and the human-written guidance strictly separate,
which makes the advice easy to update without touching the model code.

How to extend
-------------
When the actual model and dataset are confirmed, add or rename entries in
RECOMMENDATIONS to exactly match the category strings in model/labels.py.
The keys must be lowercase and match the label strings exactly.

Public API
----------
  get_recommendation(label)  → dict   Full recommendation for a label.
  get_bin_colour(label)       → str    Hex colour for that category's bin badge.
  all_categories()            → list   Sorted list of all known category keys.
"""

# ── Recommendation data ───────────────────────────────────────────────────────
# Each entry contains:
#   bin_colour   : Hex colour representing the recommended bin / stream.
#   bin_label    : Short bin description shown as a badge.
#   steps        : List of concise, actionable disposal/recycling steps.
#   tip          : One sentence of extra context or a common mistake to avoid.
#   accepted_by  : Typical collection stream (used in the UI sub-heading).
#
# NOTE: These categories are placeholders aligned to the commonly available
# TrashNet / Garbage Classification dataset labels. They WILL be updated once
# the actual dataset and trained model are confirmed. Do not treat these as
# the final label set.

RECOMMENDATIONS: dict[str, dict] = {
    "cardboard": {
        "bin_colour":   "#A0522D",   # brown — recycling / cardboard stream
        "bin_label":    "Recycling",
        "accepted_by":  "Kerbside recycling (cardboard/paper stream)",
        "steps": [
            "Flatten all cardboard boxes before placing in the bin.",
            "Remove any tape, staples, or polystyrene inserts.",
            "Keep dry — wet cardboard is rejected at most facilities.",
            "Pizza boxes with heavy grease stains belong in general waste.",
        ],
        "tip": (
            "Corrugated cardboard is one of the most recycled materials "
            "globally — flattening it saves significant space in collection vehicles."
        ),
    },

    "glass": {
        "bin_colour":   "#2E8B57",   # green — glass recycling
        "bin_label":    "Glass Recycling",
        "accepted_by":  "Glass bank / kerbside glass recycling",
        "steps": [
            "Rinse bottles and jars — remove food residue.",
            "Remove metal lids and recycle them separately.",
            "Do NOT include window glass, mirrors, or Pyrex — "
            "they melt at different temperatures and contaminate the batch.",
            "Broken glass: wrap securely in newspaper before disposal.",
        ],
        "tip": (
            "Glass can be recycled indefinitely without any loss of quality "
            "or purity — unlike most plastics."
        ),
    },

    "metal": {
        "bin_colour":   "#708090",   # slate — metal recycling
        "bin_label":    "Metal Recycling",
        "accepted_by":  "Kerbside recycling (cans/tins stream)",
        "steps": [
            "Rinse cans to remove food or drink residue.",
            "Aluminium foil and trays can be recycled — scrunch into a ball.",
            "Aerosol cans are accepted when completely empty.",
            "Sharp metal items (blades, scraps): take to a scrap metal facility.",
        ],
        "tip": (
            "Recycling one aluminium can saves enough energy to run a TV "
            "for about three hours."
        ),
    },

    "paper": {
        "bin_colour":   "#4682B4",   # steel blue — paper recycling
        "bin_label":    "Paper Recycling",
        "accepted_by":  "Kerbside recycling (paper stream)",
        "steps": [
            "Newspapers, magazines, office paper, and envelopes are all accepted.",
            "Remove plastic windows from envelopes if possible.",
            "Shredded paper: place in a paper bag before recycling.",
            "Tissues, paper towels, and waxed paper go in general waste.",
        ],
        "tip": (
            "Recycling one tonne of paper saves approximately 17 trees "
            "and 26,500 litres of water."
        ),
    },

    "plastic": {
        "bin_colour":   "#FF8C00",   # dark orange — plastic recycling
        "bin_label":    "Plastic Recycling",
        "accepted_by":  "Kerbside recycling (check local plastic grades)",
        "steps": [
            "Check the resin code (triangle number) on the bottom.",
            "Grades 1 (PET) and 2 (HDPE) are widely accepted — rinse first.",
            "Grades 3–7 acceptability varies by council — check locally.",
            "Soft plastics (bags, film, wrap) usually require a drop-off point.",
            "Do not include black plastic trays — most sorters cannot detect them.",
        ],
        "tip": (
            "Only about 9% of all plastic ever produced has been recycled. "
            "Reducing use is more impactful than recycling alone."
        ),
    },

    "trash": {
        "bin_colour":   "#696969",   # dim grey — general waste
        "bin_label":    "General Waste",
        "accepted_by":  "General waste / landfill bin",
        "steps": [
            "Place in the general waste (black/grey) bin.",
            "Check whether the item could be repaired or donated before discarding.",
            "Hazardous items (batteries, chemicals, e-waste) need special disposal.",
        ],
        "tip": (
            "When in doubt, it is better to put an item in general waste "
            "than to 'wish-cycle' — putting non-recyclables in the recycling "
            "bin contaminates entire batches."
        ),
    },
}

# ── Fallback entry ────────────────────────────────────────────────────────────
# Returned when the predicted label does not match any key above.
_UNKNOWN: dict = {
    "bin_colour":   "#B0B0B0",
    "bin_label":    "Unknown",
    "accepted_by":  "Category not yet mapped",
    "steps": [
        "This waste category is not yet in the recommendations database.",
        "Please check with your local council for disposal guidance.",
    ],
    "tip": (
        "The recommendations database will be updated once the trained "
        "model's label set is confirmed."
    ),
}


# ── Public API ────────────────────────────────────────────────────────────────

def get_recommendation(label: str) -> dict:
    """
    Return the full recommendation dict for a given waste category label.

    Parameters
    ----------
    label : The predicted category string from the model (case-insensitive).

    Returns
    -------
    A dict with keys: bin_colour, bin_label, accepted_by, steps, tip.
    If the label is not found, returns the _UNKNOWN fallback dict.
    """
    return RECOMMENDATIONS.get(label.lower().strip(), _UNKNOWN)


def get_bin_colour(label: str) -> str:
    """Return just the hex bin colour for a given label (for badge rendering)."""
    return get_recommendation(label)["bin_colour"]


def all_categories() -> list[str]:
    """Return a sorted list of all category keys currently in RECOMMENDATIONS."""
    return sorted(RECOMMENDATIONS.keys())
