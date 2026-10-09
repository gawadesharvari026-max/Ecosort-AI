"""
model/labels.py — EcoSort AI
==============================
Waste category label definitions for the TrashNet dataset.

TrashNet Overview
-----------------
TrashNet (Stanford) is a publicly available image dataset containing
approximately 2,527 images across 6 waste categories. It is one of the
most widely used benchmarks for waste classification research.

Source : https://github.com/garythung/trashnet
License: MIT

Category Distribution (original dataset)
-----------------------------------------
  cardboard : 403 images
  glass     : 501 images
  metal     : 410 images
  paper     : 594 images
  plastic   : 482 images
  trash     : 137 images

Label Convention
----------------
The model is trained with class indices assigned alphabetically by
PyTorch's ImageFolder (which sorts folder names). The mapping is:

  Index 0 → cardboard
  Index 1 → glass
  Index 2 → metal
  Index 3 → paper
  Index 4 → plastic
  Index 5 → trash

This file is the single source of truth for that mapping. Both the
training script and the inference code must import from here so that
index-to-label resolution is never duplicated or mismatched.

Public API
----------
  INDEX_TO_LABEL : dict[int, str]   Maps class index → label string.
  LABEL_TO_INDEX : dict[str, int]   Reverse mapping (label → index).
  NUM_CLASSES    : int              Total number of classes (6).
  LABEL_LIST     : list[str]        Ordered list of labels (index order).
"""

# ── Primary mapping: index → label ───────────────────────────────────────────
# Ordered alphabetically — this is the order PyTorch ImageFolder assigns
# class indices when it scans the TrashNet dataset folder structure.
INDEX_TO_LABEL: dict[int, str] = {
    0: "cardboard",
    1: "glass",
    2: "metal",
    3: "paper",
    4: "plastic",
    5: "trash",
}

# ── Reverse mapping: label → index ───────────────────────────────────────────
# Useful for mapping a string label back to its one-hot position if needed.
LABEL_TO_INDEX: dict[str, int] = {v: k for k, v in INDEX_TO_LABEL.items()}

# ── Convenience constants ─────────────────────────────────────────────────────
NUM_CLASSES: int = len(INDEX_TO_LABEL)

# Ordered list in index order — useful as the `classes` argument to
# torchvision transforms and for display in the UI.
LABEL_LIST: list[str] = [INDEX_TO_LABEL[i] for i in range(NUM_CLASSES)]


# ── Human-readable display names ─────────────────────────────────────────────
# Capitalised versions for rendering in the UI (cards, badges, tables).
DISPLAY_NAMES: dict[str, str] = {
    "cardboard": "Cardboard",
    "glass":     "Glass",
    "metal":     "Metal",
    "paper":     "Paper",
    "plastic":   "Plastic",
    "trash":     "General Trash",
}


def get_display_name(label: str) -> str:
    """
    Return the capitalised display name for a label string.

    Parameters
    ----------
    label : Lowercase label string (e.g. "cardboard").

    Returns
    -------
    Human-readable string (e.g. "Cardboard"), or the original label
    title-cased if it is not found in DISPLAY_NAMES.
    """
    return DISPLAY_NAMES.get(label.lower().strip(), label.title())


def label_from_index(index: int) -> str:
    """
    Convert a class index (from model output) to its label string.

    Parameters
    ----------
    index : Integer class index (0 – NUM_CLASSES-1).

    Returns
    -------
    Label string, or "unknown" if the index is out of range.
    """
    return INDEX_TO_LABEL.get(index, "unknown")
