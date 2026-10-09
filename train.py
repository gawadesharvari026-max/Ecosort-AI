"""
train.py — EcoSort AI
=======================
Training script for the TrashNet MobileNetV2 waste classifier.

Usage
-----
    python train.py --data-dir <path/to/trashnet/dataset-resized>

All arguments have sensible defaults — see --help for the full list.

What this script does
---------------------
1. Loads the TrashNet dataset from a folder structured as:
       <data-dir>/
           cardboard/   glass/   metal/   paper/   plastic/   trash/

2. Splits images into train / val / test sets (70 / 15 / 15 by default)
   using a reproducible random seed.

3. Applies data augmentation on the training set:
       Random horizontal flip
       Random rotation (±15°)
       Colour jitter (brightness, contrast, saturation)
       Resize → centre-crop to 224 × 224
       Normalise (ImageNet statistics)

4. Builds a MobileNetV2 backbone (pretrained on ImageNet) with the
   classification head replaced by a 6-class linear layer.

5. Trains in two phases:
   Phase 1 — freeze the backbone, train only the new head (default: 5 epochs).
              Fast convergence without overwriting ImageNet features.
   Phase 2 — unfreeze all layers, fine-tune end-to-end (default: 15 epochs).
              Backbone adapts to waste imagery with a lower learning rate.

6. After each epoch:
       Prints training loss and validation accuracy.
       Saves the model weights if validation accuracy improved
       (early-stopping via best-model checkpointing).

7. After training completes:
       Evaluates on the held-out test set.
       Prints per-class precision, recall, F1, and support.
       Prints overall accuracy.
       Saves final weights to model/trashnet_mobilenetv2.pth.

Output
------
The trained weights are saved as a state_dict:
    torch.save(model.state_dict(), "model/trashnet_mobilenetv2.pth")

This matches exactly what model/classifier.py expects to load.

Requirements
------------
    torch >= 2.0
    torchvision >= 0.15
    scikit-learn          (for classification_report)
    Pillow                (for image loading)
    numpy

These are already installed in the project environment.
"""

import argparse
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from model.labels import NUM_CLASSES, LABEL_LIST


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducible train/val/test splits and weight init."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ── Image transforms ──────────────────────────────────────────────────────────
# Training transform includes augmentation to reduce overfitting on the small
# TrashNet dataset (~2,527 images total).
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),   # random zoom + crop
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Validation and test transforms — deterministic, no augmentation.
# Must match the preprocessing in model/classifier.py exactly.
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Dataset splitting ─────────────────────────────────────────────────────────
def make_splits(
    data_dir: str,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> tuple[Subset, Subset, Subset]:
    """
    Load the full dataset with ImageFolder and split into train / val / test.

    ImageFolder expects sub-folders named after each class:
        <data-dir>/cardboard/  <data-dir>/glass/  … etc.

    The alphabetical folder order defines the class-to-index mapping, which
    must match INDEX_TO_LABEL in model/labels.py.

    Parameters
    ----------
    data_dir  : Path to the dataset root directory.
    val_size  : Fraction of data for validation (default 0.15).
    test_size : Fraction of data for test (default 0.15).
    seed      : Random seed for reproducibility.

    Returns
    -------
    Three Subset objects: train_set, val_set, test_set.
    Each wraps the same underlying ImageFolder with independent transforms
    applied via a thin wrapper class.
    """
    # Load full dataset (transform=None — we apply per-split transforms below)
    full_dataset = datasets.ImageFolder(data_dir)

    # Verify the class-to-index mapping matches our label definitions
    folder_classes = full_dataset.classes   # alphabetically sorted
    expected_classes = sorted(LABEL_LIST)
    if folder_classes != expected_classes:
        print(
            f"\n⚠️  WARNING: Dataset folder names do not match model/labels.py.\n"
            f"   Found in {data_dir}: {folder_classes}\n"
            f"   Expected:              {expected_classes}\n"
            f"   Class indices will follow the folder names found on disk.\n"
            f"   Update model/labels.py if they differ.\n"
        )

    indices = list(range(len(full_dataset)))
    labels  = [full_dataset.targets[i] for i in indices]

    # First split: train vs (val + test) — stratified to keep class balance
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(val_size + test_size),
        stratify=labels,
        random_state=seed,
    )

    temp_labels = [labels[i] for i in temp_idx]
    relative_test = test_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test,
        stratify=temp_labels,
        random_state=seed,
    )

    # Wrap with per-split transforms
    train_set = _TransformSubset(full_dataset, train_idx, TRAIN_TRANSFORM)
    val_set   = _TransformSubset(full_dataset, val_idx,   EVAL_TRANSFORM)
    test_set  = _TransformSubset(full_dataset, test_idx,  EVAL_TRANSFORM)

    return train_set, val_set, test_set


class _TransformSubset(Subset):
    """
    A Subset that applies a transform when items are accessed.

    PyTorch's built-in Subset does not support per-split transforms because
    the underlying Dataset holds a single transform. This subclass overrides
    __getitem__ to apply the transform at access time instead.
    """
    def __init__(self, dataset, indices, transform):
        super().__init__(dataset, indices)
        self._transform = transform

    def __getitem__(self, idx):
        image, label = self.dataset.samples[self.indices[idx]]
        image = Image.open(image).convert("RGB")
        if self._transform:
            image = self._transform(image)
        return image, label
        
    def __getitems__(self, indices):
        return [self.__getitem__(idx) for idx in indices]

# ── Model construction ────────────────────────────────────────────────────────
def build_model(pretrained: bool = True) -> nn.Module:
    """
    Build a MobileNetV2 with a 6-class classification head.

    This MUST produce the same architecture as model/classifier.py:_build_model()
    so the saved weights load correctly at inference time.

    Parameters
    ----------
    pretrained : If True (default), load ImageNet pretrained weights for the
                 backbone. Set to False only for unit tests.
    """
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    # Replace the 1000-class head with a 6-class head
    in_features = model.classifier[1].in_features   # 1280
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

    return model


def freeze_backbone(model: nn.Module) -> None:
    """
    Freeze all layers except the classifier head.
    Used in Phase 1 to train only the new head.
    """
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("classifier")


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all parameters for Phase 2 (full fine-tuning)."""
    for param in model.parameters():
        param.requires_grad = True


# ── Training and evaluation loops ────────────────────────────────────────────
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Run one training epoch.

    Returns
    -------
    Average cross-entropy loss over all batches.
    """
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, list[int], list[int]]:
    """
    Evaluate the model on a DataLoader.

    Returns
    -------
    accuracy     : Float 0.0–1.0.
    all_preds    : List of predicted class indices.
    all_targets  : List of ground-truth class indices.
    """
    model.eval()
    correct = 0
    total   = 0
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds   = outputs.argmax(dim=1)

            correct      += (preds == labels).sum().item()
            total        += labels.size(0)
            all_preds    += preds.cpu().tolist()
            all_targets  += labels.cpu().tolist()

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, all_preds, all_targets


# ── Main training routine ─────────────────────────────────────────────────────
def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = torch.device("cpu")   # CPU-only environment
    print(f"\n{'─'*60}")
    print(f"  EcoSort AI — TrashNet MobileNetV2 Training")
    print(f"{'─'*60}")
    print(f"  Data directory : {args.data_dir}")
    print(f"  Phase 1 epochs : {args.phase1_epochs}  (head only)")
    print(f"  Phase 2 epochs : {args.phase2_epochs}  (full fine-tune)")
    print(f"  Batch size     : {args.batch_size}")
    print(f"  Output path    : {args.output}")
    print(f"  Device         : {device}")
    print(f"{'─'*60}\n")

    # ── Validate data directory ───────────────────────────────────────────────
    if not os.path.isdir(args.data_dir):
        print(
            f"❌ Data directory not found: {args.data_dir}\n\n"
            "   Please download the TrashNet dataset first:\n"
            "     git clone https://github.com/garythung/trashnet.git\n"
            "   Then pass the dataset-resized folder:\n"
            "     python train.py --data-dir trashnet/dataset-resized\n"
        )
        return

    # ── Build data splits ─────────────────────────────────────────────────────
    print("📂 Loading dataset and creating splits…")
    train_set, val_set, test_set = make_splits(
        args.data_dir,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    print(
        f"   Train: {len(train_set)} | "
        f"Val: {len(val_set)} | "
        f"Test: {len(test_set)} images\n"
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,    # 0 is safest on Windows with Streamlit's multiprocessing
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # ── Build model ───────────────────────────────────────────────────────────
    print("🧠 Building MobileNetV2 (ImageNet pretrained backbone)…")
    model = build_model(pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Best-model checkpoint — saves weights when val accuracy improves
    best_val_acc  = 0.0
    best_weights  = None
    output_path   = args.output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # ── Phase 1: train head only ──────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Phase 1 — Training classifier head only ({args.phase1_epochs} epochs)")
    print(f"{'─'*60}")

    freeze_backbone(model)
    # Only pass parameters that require gradients to the optimiser
    optimizer_p1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_phase1,
    )
    scheduler_p1 = optim.lr_scheduler.StepLR(optimizer_p1, step_size=3, gamma=0.5)

    for epoch in range(1, args.phase1_epochs + 1):
        t0       = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer_p1, device)
        val_acc, _, _ = evaluate(model, val_loader, device)
        scheduler_p1.step()
        elapsed  = time.time() - t0

        marker = " ✅ (best)" if val_acc > best_val_acc else ""
        print(
            f"  Epoch {epoch:02d}/{args.phase1_epochs:02d}  "
            f"loss={train_loss:.4f}  val_acc={val_acc*100:.1f}%  "
            f"({elapsed:.0f}s){marker}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_weights, output_path)

    # ── Phase 2: full fine-tuning ─────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Phase 2 — Full fine-tuning ({args.phase2_epochs} epochs, lr={args.lr_phase2})")
    print(f"{'─'*60}")

    unfreeze_all(model)
    optimizer_p2 = optim.Adam(model.parameters(), lr=args.lr_phase2)
    scheduler_p2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p2, T_max=args.phase2_epochs
    )

    for epoch in range(1, args.phase2_epochs + 1):
        t0         = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer_p2, device)
        val_acc, _, _ = evaluate(model, val_loader, device)
        scheduler_p2.step()
        elapsed    = time.time() - t0

        marker = " ✅ (best)" if val_acc > best_val_acc else ""
        print(
            f"  Epoch {epoch:02d}/{args.phase2_epochs:02d}  "
            f"loss={train_loss:.4f}  val_acc={val_acc*100:.1f}%  "
            f"({elapsed:.0f}s){marker}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_weights, output_path)

    # ── Test set evaluation ───────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Test set evaluation (best checkpoint: val_acc={best_val_acc*100:.1f}%)")
    print(f"{'─'*60}")

    # Load the best checkpoint before evaluating on the test set
    if best_weights is not None:
        model.load_state_dict(best_weights)

    test_acc, test_preds, test_targets = evaluate(model, test_loader, device)
    print(f"\n  Test accuracy: {test_acc*100:.2f}%\n")

    # Per-class report using sklearn — shows real precision, recall, F1, support
    class_names = [c.capitalize() for c in LABEL_LIST]
    report = classification_report(
        test_targets,
        test_preds,
        target_names=class_names,
        digits=3,
    )
    print("  Per-class metrics:")
    for line in report.splitlines():
        print(f"    {line}")

    # ── Save final weights ────────────────────────────────────────────────────
    torch.save(best_weights if best_weights else model.state_dict(), output_path)
    print(f"\n✅ Best model weights saved to: {output_path}")
    print(
        f"\n   Run the app:  streamlit run app.py\n"
        f"   The 'model not configured' notice will be gone.\n"
    )


# ── CLI argument parser ───────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a MobileNetV2 classifier on the TrashNet dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="trashnet/dataset-resized",
        help="Path to the TrashNet dataset root (contains cardboard/, glass/, … sub-folders).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model/trashnet_mobilenetv2.pth",
        help="Where to save the best model weights (.pth file).",
    )
    parser.add_argument(
        "--phase1-epochs",
        type=int,
        default=5,
        dest="phase1_epochs",
        help="Epochs to train only the classification head (backbone frozen).",
    )
    parser.add_argument(
        "--phase2-epochs",
        type=int,
        default=15,
        dest="phase2_epochs",
        help="Epochs for full end-to-end fine-tuning.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        dest="batch_size",
        help="Images per batch.",
    )
    parser.add_argument(
        "--lr-phase1",
        type=float,
        default=1e-3,
        dest="lr_phase1",
        help="Learning rate for Phase 1 (head-only).",
    )
    parser.add_argument(
        "--lr-phase2",
        type=float,
        default=1e-4,
        dest="lr_phase2",
        help="Learning rate for Phase 2 (full fine-tune). Should be lower than Phase 1.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        dest="val_size",
        help="Fraction of data held out for validation.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        dest="test_size",
        help="Fraction of data held out for final test evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits.",
    )

    return parser.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    train(args)
