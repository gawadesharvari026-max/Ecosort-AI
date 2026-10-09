"""
pages/03_About.py — EcoSort AI
=================================
About page.

Covers:
  - What EcoSort AI is and its purpose
  - The TrashNet dataset (categories, counts, source)
  - Model architecture (MobileNetV2) and why it was chosen
  - Honest limitations and disclaimers — no claimed accuracy figures
  - How to set up and train the model (step-by-step instructions)
  - Technology stack
  - Project credits
"""

import streamlit as st

from components.ui_helpers import (
    render_page_header,
    render_divider,
    render_notice,
    render_warning,
)
from model.classifier import is_model_available, MODEL_PATH
from model.labels import LABEL_LIST, INDEX_TO_LABEL
from data.recommendations import all_categories

# ── Page header ───────────────────────────────────────────────────────────────
render_page_header(
    "ℹ️ About EcoSort AI",
    "Project overview, dataset information, model architecture, and setup guide.",
)

# ── Model status banner ───────────────────────────────────────────────────────
if is_model_available():
    render_notice(
        f"<strong>Model status: Ready.</strong> "
        f"Weights file found at <code>{MODEL_PATH}</code>.",
        icon="✅",
    )
else:
    render_warning(
        f"<strong>Model status: Not configured.</strong> "
        f"No weights file found at <code>{MODEL_PATH}</code>. "
        f"Classification is disabled until the model is trained and the file is placed there.",
        icon="⚠️",
    )

render_divider()

# ── Project overview ──────────────────────────────────────────────────────────
st.markdown("### 🌱 What is EcoSort AI?")
st.markdown(
    """
    **EcoSort AI** is an intelligent waste classification and recycling recommendation
    system built as a Machine Learning and Applied AI internship project.

    It uses a deep learning image classifier to identify the waste category of an
    uploaded photo, then provides practical recycling and disposal guidance based on
    that category.

    The system is designed to be:
    - **Honest** — predictions are only shown when a real trained model is available.
      No fake confidence scores or placeholder results are ever displayed.
    - **Modular** — the ML classifier, the recycling recommendations, and the
      classification history are kept in separate, independently testable modules.
    - **Beginner-friendly** — every module is documented with clear explanations of
      why design decisions were made.
    """
)

render_divider()

# ── Dataset ───────────────────────────────────────────────────────────────────
st.markdown("### 📦 Dataset — TrashNet")

col_info, col_table = st.columns([3, 2], gap="large")

with col_info:
    st.markdown(
        """
        EcoSort AI is designed to train on **TrashNet**, a publicly available image
        dataset created at Stanford University.

        - **Source:** [github.com/garythung/trashnet](https://github.com/garythung/trashnet)
        - **Licence:** MIT
        - **Total images:** ~2,527
        - **Image size:** 512 × 384 px (JPEG)
        - **Classes:** 6 waste categories

        The dataset is **not bundled** with this application. You must download it
        separately before training the model.
        """
    )

with col_table:
    st.markdown(
        """
        <div class='eco-card'>
        <div style='font-size:0.78rem; color:#57606a; margin-bottom:0.6rem;
                    text-transform:uppercase; letter-spacing:0.06em;'>
        Class distribution
        </div>

        | Category | Images |
        |---|---|
        | Cardboard | 403 |
        | Glass | 501 |
        | Metal | 410 |
        | Paper | 594 |
        | Plastic | 482 |
        | Trash | 137 |
        | **Total** | **2,527** |

        </div>
        """,
        unsafe_allow_html=True,
    )

render_notice(
    "The class distribution above is from the original TrashNet dataset. "
    "Actual training results — including accuracy, precision, and recall — "
    "will be reported here once the model has been trained. "
    "<strong>No accuracy statistics are claimed for an untrained model.</strong>",
    icon="📋",
)

render_divider()

# ── Model architecture ────────────────────────────────────────────────────────
st.markdown("### 🧠 Model Architecture — MobileNetV2")
st.markdown(
    """
    EcoSort AI uses a **MobileNetV2** convolutional neural network, loaded from
    `torchvision.models`, with **transfer learning**:

    1. **Backbone** — MobileNetV2 pretrained on ImageNet (1.2 million images,
       1,000 classes). This gives the model strong general visual features
       (edges, textures, shapes) without needing a large dataset.

    2. **Custom classification head** — the original 1,000-class output layer is
       replaced with a 6-class fully-connected layer, one output per TrashNet category.

    3. **Fine-tuning** — during training, the new head is trained first; then the
       backbone weights are gradually unfrozen to adapt to waste imagery.

    **Why MobileNetV2?**
    - Designed for CPU-bound environments — no GPU required.
    - Strong accuracy on small transfer-learning datasets.
    - Ships with `torchvision` — no separate download needed.

    **Input preprocessing** (applied to every image before inference):
    - Resize shortest side to 256 px
    - Centre-crop to 224 × 224 px
    - Normalise with ImageNet channel statistics
      (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`)
    """
)

render_divider()

# ── Limitations and honest disclaimers ───────────────────────────────────────
st.markdown("### ⚠️ Limitations and Disclaimers")

render_warning(
    """
    <strong>The model has not been trained yet.</strong> No accuracy, precision,
    recall, or F1 scores are available. These will be reported here once the
    model has been trained on the TrashNet dataset.<br><br>
    Even after training, please be aware:
    <ul style='margin:0.5rem 0 0 1rem;'>
      <li>Image classification models can and do make mistakes.</li>
      <li>A single photo may not capture enough detail for a confident prediction.</li>
      <li>Waste disposal rules vary by region — always verify with your local council.</li>
      <li>The <em>Recycling &amp; Disposal Guide</em> tab contains general guidance only,
          not legal or regulatory advice.</li>
    </ul>
    """,
    icon="⚠️",
)

render_divider()

# ── Setup guide ───────────────────────────────────────────────────────────────
st.markdown("### 🛠️ How to Enable Classification")
st.markdown(
    """
    Follow these steps to go from the current skeleton to a working classifier:

    **Step 1 — Download the TrashNet dataset**
    ```
    git clone https://github.com/garythung/trashnet.git
    ```
    The images are inside `trashnet/dataset-resized/`. Each sub-folder name
    is a category label (cardboard, glass, metal, paper, plastic, trash).

    **Step 2 — Organise the dataset**
    Split the images into `train/`, `val/`, and `test/` sub-folders, each
    containing one folder per class. A 70 / 15 / 15 split is typical.

    **Step 3 — Train the model** *(training script to be added in a later step)*
    ```
    python train.py --data-dir ./dataset --epochs 20 --output model/trashnet_mobilenetv2.pth
    ```

    **Step 4 — Place the weights file**
    Copy the output `.pth` file to:
    ```
    model/trashnet_mobilenetv2.pth
    ```

    **Step 5 — Restart the app**
    ```
    streamlit run app.py
    ```
    The model-not-configured notice will disappear and the Classify Waste
    page will be fully functional.
    """
)

render_divider()

# ── Technology stack ──────────────────────────────────────────────────────────
st.markdown("### 🔧 Technology Stack")

col_a, col_b = st.columns(2, gap="large")

with col_a:
    st.markdown(
        """
        <div class='eco-card'>
        <strong>Application layer</strong>
        <hr class='eco-divider'>

        | Component | Technology |
        |---|---|
        | Web framework | Streamlit 1.65 |
        | Image display | Pillow |
        | Charts | Plotly |
        | History storage | SQLite (stdlib) |
        | Data handling | Pandas, NumPy |

        </div>
        """,
        unsafe_allow_html=True,
    )

with col_b:
    st.markdown(
        """
        <div class='eco-card'>
        <strong>Machine learning layer</strong>
        <hr class='eco-divider'>

        | Component | Technology |
        |---|---|
        | Framework | PyTorch 2.14.1+cpu |
        | Model | MobileNetV2 (torchvision) |
        | Dataset | TrashNet (MIT licence) |
        | Task | Image classification |
        | Classes | 6 waste categories |

        </div>
        """,
        unsafe_allow_html=True,
    )

render_divider()

# ── Project credits ───────────────────────────────────────────────────────────
st.markdown("### 📄 Credits")
st.markdown(
    """
    - **TrashNet dataset** — Gary Thung & Mindy Yang, Stanford University
      ([MIT Licence](https://github.com/garythung/trashnet/blob/master/LICENSE))
    - **MobileNetV2** — Sandler et al., Google (2018)
    - **PyTorch** — Meta AI Research
    - **Streamlit** — Streamlit Inc.
    - **EcoSort AI** — Machine Learning & Applied AI Internship Project
    """
)

render_divider()
st.caption(
    "EcoSort AI — Built with Streamlit and PyTorch · "
    "Machine Learning & Applied AI Internship Project"
)
