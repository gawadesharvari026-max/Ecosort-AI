"""
model/classifier.py — EcoSort AI
===================================
PyTorch-based image classifier for TrashNet waste categories.

Architecture
------------
Uses a MobileNetV2 backbone (pretrained on ImageNet) with its final
classification head replaced by a 6-output linear layer — one output
per TrashNet category. MobileNetV2 is chosen because:
  - It is lightweight and runs well on CPU (no GPU required).
  - torchvision bundles it with pretrained ImageNet weights.
  - It consistently achieves strong accuracy on small datasets
    like TrashNet when fine-tuned.

Model File
----------
The trained model weights must be saved as:

    model/trashnet_mobilenetv2.pth

This file is NOT included in the repository. You must train the model
first (a training script will be provided separately) or obtain a
compatible .pth file and place it at the path above.

Until that file exists, all calls to classify_image() will raise a
ModelNotConfiguredError, which the UI catches and displays as a
friendly notice — no fake predictions are ever returned.

Preprocessing
-------------
All input images are resized and normalised to match the ImageNet
statistics used during MobileNetV2 pretraining, which is standard
practice for transfer learning:
  Resize  : 256 × 256, then centre-crop to 224 × 224
  Mean    : [0.485, 0.456, 0.406]
  Std     : [0.229, 0.224, 0.225]

Public API
----------
  load_model()                     → nn.Module   Load weights from disk.
  classify_image(pil_image)        → ClassificationResult
  is_model_available()             → bool        Check if .pth file exists.

  ClassificationResult (dataclass):
    label        str    Predicted category (e.g. "plastic")
    display_name str    Human-readable label (e.g. "Plastic")
    confidence   float  Softmax probability of the top class (0.0–1.0)
    all_scores   dict   {label: probability} for all 6 classes
"""

import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from model.labels import (
    NUM_CLASSES,
    LABEL_LIST,
    label_from_index,
    get_display_name,
)

# ── Model file path ───────────────────────────────────────────────────────────
# Expected location: model/trashnet_mobilenetv2.pth
# (relative to the project root, i.e. one level up from this file)
_MODEL_DIR  = os.path.dirname(__file__)
MODEL_PATH  = os.path.join(_MODEL_DIR, "trashnet_mobilenetv2.pth")


# ── Custom exception ──────────────────────────────────────────────────────────
class ModelNotConfiguredError(Exception):
    """
    Raised when classify_image() is called but no model file has been placed
    at MODEL_PATH. The Streamlit UI catches this and shows a friendly notice
    instead of a Python traceback.
    """


# ── Inference result ──────────────────────────────────────────────────────────
@dataclass
class ClassificationResult:
    """
    Holds the output of a single classification call.

    Attributes
    ----------
    label        : Lowercase category string (e.g. "plastic").
    display_name : Capitalised UI-friendly name (e.g. "Plastic").
    confidence   : Softmax probability of the predicted class, 0.0 – 1.0.
    all_scores   : Dict mapping every label to its softmax probability.
                   Values sum to ~1.0 and are useful for a confidence bar chart.
    """
    label:        str
    display_name: str
    confidence:   float
    all_scores:   dict[str, float]


# ── Image preprocessing pipeline ─────────────────────────────────────────────
# Matches the preprocessing used during MobileNetV2 ImageNet pretraining.
# Must be applied to every image before passing it through the model.
_PREPROCESS = transforms.Compose([
    transforms.Resize(256),           # Scale shortest side to 256 px
    transforms.CenterCrop(224),       # Crop to 224×224 (MobileNetV2 input size)
    transforms.ToTensor(),            # PIL Image → [0,1] float tensor (C, H, W)
    transforms.Normalize(             # Normalise with ImageNet channel statistics
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def _build_model() -> nn.Module:
    """
    Construct a MobileNetV2 with its classifier head replaced for NUM_CLASSES.

    This must produce the exact same architecture that was used during
    training — if the architecture changes, the saved weights will not load.

    The final classifier of MobileNetV2 is:
        model.classifier[1]  →  nn.Linear(1280, num_classes)
    """
    # Load MobileNetV2 architecture (no pretrained weights — we load ours below)
    model = models.mobilenet_v2(weights=None)

    # Replace the default 1000-class head with a 6-class head for TrashNet
    in_features = model.classifier[1].in_features   # 1280
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

    return model


def is_model_available() -> bool:
    """
    Return True if the trained model weights file exists at MODEL_PATH.
    Used by the UI to decide whether to show the 'not configured' notice.
    """
    return os.path.isfile(MODEL_PATH)


def load_model() -> nn.Module:
    """
    Load the trained MobileNetV2 weights from MODEL_PATH.

    Returns
    -------
    A PyTorch nn.Module in evaluation mode, ready for inference.

    Raises
    ------
    ModelNotConfiguredError
        If MODEL_PATH does not exist.
    RuntimeError
        If the weights file exists but cannot be loaded (e.g. corrupted file
        or architecture mismatch). The original PyTorch error is chained.
    """
    if not is_model_available():
        raise ModelNotConfiguredError(
            f"Model weights not found at: {MODEL_PATH}\n\n"
            "To enable classification:\n"
            "  1. Download the TrashNet dataset.\n"
            "  2. Run the training script to produce trashnet_mobilenetv2.pth.\n"
            "  3. Place the .pth file in the model/ folder.\n"
            "  4. Restart the app.\n\n"
            "No predictions will be shown until the model file is present."
        )

    model = _build_model()

    try:
        # map_location="cpu" ensures the weights load even if they were saved
        # on a GPU machine — this app targets CPU-only environments.
        state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model weights from {MODEL_PATH}.\n"
            "The file may be corrupted or saved with a different architecture.\n"
            f"Original error: {exc}"
        ) from exc

    model.eval()   # Disable dropout / batch-norm training behaviour
    return model


def classify_image(
    pil_image: Image.Image,
    model: nn.Module | None = None,
) -> ClassificationResult:
    """
    Run inference on a single PIL Image and return a ClassificationResult.

    Parameters
    ----------
    pil_image : A PIL.Image.Image object (any mode — converted to RGB internally).
    model     : Optional pre-loaded model. If None, load_model() is called
                automatically. Pass a cached model from Streamlit's
                st.cache_resource to avoid reloading on every call.

    Returns
    -------
    ClassificationResult with label, display_name, confidence, all_scores.

    Raises
    ------
    ModelNotConfiguredError
        If no model file exists (propagated from load_model()).
    ValueError
        If pil_image is not a PIL Image object.
    """
    if not isinstance(pil_image, Image.Image):
        raise ValueError(
            f"Expected a PIL.Image.Image, got {type(pil_image).__name__}."
        )

    # Load model if not provided
    if model is None:
        model = load_model()

    # Convert to RGB — handles RGBA, greyscale, palette-mode images etc.
    image_rgb = pil_image.convert("RGB")

    # Preprocess: resize → crop → tensor → normalise
    input_tensor = _PREPROCESS(image_rgb)          # shape: (3, 224, 224)
    input_batch  = input_tensor.unsqueeze(0)        # shape: (1, 3, 224, 224)

    # Run inference — no gradient computation needed for inference
    with torch.no_grad():
        logits = model(input_batch)                 # shape: (1, NUM_CLASSES)

    # Convert raw logits → probabilities via softmax
    probabilities = torch.softmax(logits, dim=1).squeeze(0)  # shape: (NUM_CLASSES,)

    # Identify the top prediction
    top_index      = int(probabilities.argmax())
    top_confidence = float(probabilities[top_index])
    top_label      = label_from_index(top_index)

    # Build the per-class score dict for the confidence chart
    all_scores = {
        LABEL_LIST[i]: float(probabilities[i])
        for i in range(NUM_CLASSES)
    }

    return ClassificationResult(
        label        = top_label,
        display_name = get_display_name(top_label),
        confidence   = top_confidence,
        all_scores   = all_scores,
    )
