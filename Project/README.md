# M7016H Final Project: Breast Ultrasound Image Classification
### AI-Based Medical Decision Support System for Breast Cancer Detection
**Course:** M7016H – Artificiell intelligens inom sjukvården  
**Dataset:** Breast Ultrasound Images (Al-Dhabyani et al., 2020)  
**Task:** Multi-class classification — Normal / Benign / Malignant  
**Deadlines:**
- 15 May 2026 @ 23:59 → Report (PDF) + Video + Slides (PDF) + Code
- 26 May 2026 @ 23:59 → Reflection (PDF, post-symposium)

---

## Table of Contents
1. [Clinical Motivation](#1-clinical-motivation)
2. [Dataset Summary](#2-dataset-summary)
3. [Project Folder Structure](#3-project-folder-structure)
4. [Environment Setup](#4-environment-setup)
5. [Data Split Strategy — Train / Validation / Test](#5-data-split-strategy)
6. [Preprocessing Pipeline](#6-preprocessing-pipeline)
7. [Data Augmentation](#7-data-augmentation)
8. [Handling Class Imbalance](#8-handling-class-imbalance)
9. [Model Architecture](#9-model-architecture)
10. [Hyperparameter Optimization with Validation Set](#10-hyperparameter-optimization)
11. [Training Loop](#11-training-loop)
12. [Evaluation & Performance Metrics](#12-evaluation--performance-metrics)
13. [Explainability — Grad-CAM](#13-explainability--grad-cam)
14. [Figures & Tables for Report](#14-figures--tables-for-report)
15. [Report Outline (3 pages)](#15-report-outline)
16. [Flash Talk Slide Structure](#16-flash-talk-slide-structure)
17. [AI Use Documentation Template](#17-ai-use-documentation-template)
18. [Day-by-Day Schedule](#18-day-by-day-schedule)
19. [Submission Checklist](#19-submission-checklist)
20. [Common Pitfalls](#20-common-pitfalls)
21. [Key References](#21-key-references)

---

## 1. Clinical Motivation

Breast cancer is the most frequently diagnosed cancer among women globally. Early detection
is the single most impactful factor in reducing mortality. Ultrasound (US) is a widely used,
radiation-free, low-cost modality especially valuable for younger patients or dense breast tissue
where mammography is less sensitive. However, interpretation is operator-dependent and
radiologist availability varies. An AI-based decision support system that classifies US images
into **normal**, **benign**, and **malignant** categories could:

- Act as a reliable second reader, reducing missed diagnoses
- Support radiologists in high-volume or resource-limited settings
- Improve consistency and reduce inter-reader variability

This forms the core clinical justification for your report Introduction section.

---

## 2. Dataset Summary

| Class     | Total Images | Train (70%) | Val (15%) | Test (15%) |
|-----------|:------------:|:-----------:|:---------:|:----------:|
| Benign    | 437          | ~306        | ~65       | ~66        |
| Malignant | 210          | ~147        | ~32       | ~31        |
| Normal    | 133          | ~93         | ~20       | ~20        |
| **Total** | **780**      | **~546**    | **~117**  | **~117**   |

> ⚠️ **Important:** Each image has a paired `_mask.png` file in the same folder.
> These mask files must be **explicitly excluded** from all splits.
> Exact counts after filtering masks will differ slightly.

- **Format:** PNG, grayscale, ~500×500 px average
- **Source:** Baheya Hospital, Cairo, Egypt (2018); 600 female patients, ages 25–75
- **No pre-made split is provided for Dataset 3** — you must create your own
- **Citation required:** Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. *Data in Brief.* 2020;28:104863.

---

## 3. Project Folder Structure

```
project/
├── data/
│   ├── raw/                         # Original downloaded dataset (do NOT modify)
│   │   ├── benign/
│   │   │   ├── benign (1).png
│   │   │   ├── benign (1)_mask.png   ← excluded these!
│   │   │   └── ...
│   │   ├── malignant/
│   │   └── normal/
│   └── processed/                   # Output of preprocessing script
│       ├── train/
│       │   ├── benign/
│       │   ├── malignant/
│       │   └── normal/
│       ├── val/                     ← VALIDATION SET (used during training)
│       │   ├── benign/
│       │   ├── malignant/
│       │   └── normal/
│       └── test/                    ← HELD-OUT TEST SET (touched only at the end)
│           ├── benign/
│           ├── malignant/
│           └── normal/
│
├── notebooks/
│   ├── 01_EDA.ipynb                 # Exploratory data analysis
│   ├── 02_Preprocessing.ipynb       # Split + preprocessing pipeline
│   ├── 03_Model_Training.ipynb      # Training with validation monitoring
│   ├── 04_Hyperparameter_Tuning.ipynb  # Hyperparameter search using val set
│   ├── 05_Evaluation.ipynb          # Final test set evaluation
│   └── 06_GradCAM_Explainability.ipynb
│
├── src/
│   ├── dataset.py                   # PyTorch Dataset class
│   ├── model.py                     # Model architecture definition
│   ├── train.py                     # Training + validation loop
│   ├── evaluate.py                  # Metrics: sensitivity, specificity, AUROC, F1
│   └── utils.py                     # Plotting, early stopping, seeds
│
├── outputs/
│   ├── models/
│   │   ├── best_model.pth           # Best checkpoint (by val loss)
│   │   └── final_model.pth
│   ├── figures/
│   │   ├── fig1_sample_images.png
│   │   ├── fig2_class_distribution.png
│   │   ├── fig3_training_curves.png
│   │   ├── fig4_confusion_matrix.png
│   │   └── fig5_roc_curves.png
│   └── logs/
│       ├── training_log.csv          # epoch, train_loss, val_loss, val_acc
│       └── hyperparam_results.csv    # results from tuning experiments
│
├── report/
│   └── report.pdf
│
├── presentation/
│   ├── slides.pptx
│   └── slides.pdf
│
├── requirements.txt
├── README.md
└── AI_use_documentation.md          # Required by LTU AI policy
```

---

## 4. Environment Setup

### Option A: Local (recommended if you have a GPU)
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### Option B: Google Colab (free GPU — recommended if no local GPU)
```python
# At the top of your first Colab notebook:
!pip install grad-cam torch torchvision scikit-learn matplotlib seaborn tqdm

# Mount Google Drive to persist your data and checkpoints
from google.colab import drive
drive.mount('/content/drive')
```

### requirements.txt
```
torch>=2.0.0
torchvision>=0.15.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
Pillow>=9.0.0
numpy>=1.23.0
pandas>=1.5.0
tqdm>=4.64.0
grad-cam>=1.4.6
jupyter>=1.0.0
```

---

## 5. Data Split Strategy

### Why three splits?

| Split | Purpose | When used |
|-------|---------|-----------|
| **Train (70%)** | Model learns weights from this data | Every epoch during training |
| **Validation (15%)** | Monitor overfitting; tune hyperparameters; pick best model checkpoint | After every epoch; during hyperparameter search |
| **Test (15%)** | Final, unbiased evaluation reported in the paper | **Only once**, after all tuning is done |

> ⚠️ **Critical rule:** The test set is a "time capsule." You must not look at test
> results, adjust your model, and then re-test. That would be data leakage.
> Tune everything using the **validation set**. Report final performance on the **test set**.

### Splitting code (stratified, reproducible)
```python
import os, shutil, random
from pathlib import Path
from collections import defaultdict

# Set seed for reproducibility — always use the same seed
SEED = 42
random.seed(SEED)

SRC  = Path("data/raw")
DEST = Path("data/processed")

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15  # = 1 - TRAIN_RATIO - VAL_RATIO

split_counts = defaultdict(lambda: defaultdict(int))

for cls in ["benign", "malignant", "normal"]:
    # Filter out mask images — critical step!
    all_imgs = sorted([
        f for f in (SRC / cls).glob("*.png")
        if "_mask" not in f.name
    ])

    random.shuffle(all_imgs)
    n = len(all_imgs)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    # Test gets the remainder to avoid rounding errors dropping images
    n_test  = n - n_train - n_val

    splits = {
        "train": all_imgs[:n_train],
        "val":   all_imgs[n_train : n_train + n_val],
        "test":  all_imgs[n_train + n_val :]
    }

    for split_name, imgs in splits.items():
        out_dir = DEST / split_name / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for img_path in imgs:
            shutil.copy(img_path, out_dir / img_path.name)
        split_counts[split_name][cls] = len(imgs)
        print(f"  {split_name:6s}/{cls:10s}: {len(imgs)} images")

# Verify totals
print("\n--- Split Summary ---")
for split_name in ["train", "val", "test"]:
    total = sum(split_counts[split_name].values())
    print(f"{split_name}: {total} images total")
```

**Expected output:**
```
  train /benign    : 306 images
  train /malignant : 147 images
  train /normal    :  93 images
  val   /benign    :  65 images
  val   /malignant :  32 images
  val   /normal    :  20 images
  test  /benign    :  66 images
  test  /malignant :  31 images
  test  /normal    :  20 images

--- Split Summary ---
train: 546 images total
val  : 117 images total
test : 117 images total
```

---

## 6. Preprocessing Pipeline

All images must be preprocessed identically for train, val, and test sets,
**except** that augmentation is applied only to the training set.

```python
from torchvision import transforms

IMG_SIZE = 224  # ResNet-18 input size

# Normalization stats — using ImageNet values since we use pretrained weights
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Applied to TRAIN only — augmentation helps generalization on small dataset
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),   # US is grayscale; ResNet needs 3ch
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),         # Flipping valid for US images
    transforms.RandomRotation(degrees=15),           # Small rotation mimics probe angle variation
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Simulate scanner differences
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# Applied to VALIDATION and TEST — no augmentation, just normalization
val_test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])
```

> **Justification to mention in report:**
> - Resize to 224×224: required by ResNet-18 architecture
> - Grayscale→3-channel: pretrained ResNet expects 3-channel input; replicating the single
>   channel across 3 channels preserves spatial information
> - ImageNet normalization: used because we initialize with ImageNet pretrained weights;
>   the model's internal feature detectors expect this input distribution
> - Augmentation on train only: val/test must reflect real-world conditions — augmenting
>   them would give an unrealistic (optimistic) view of performance

---

## 7. Data Augmentation

Augmentation artificially expands the effective training set and reduces overfitting.
Choices must be clinically plausible — transformations that could realistically
appear in practice.

| Augmentation | Justification |
|-------------|---------------|
| Horizontal flip | Lesions can appear on either side of an image |
| Rotation ±15° | Probe orientation varies slightly between scans |
| Brightness/contrast jitter ±20% | Different ultrasound machines produce varying image quality |
| **NOT used:** vertical flip | Anatomically unrealistic for breast US |
| **NOT used:** heavy crop/warp | Could remove the lesion from the image |

---

## 8. Handling Class Imbalance

The dataset is heavily imbalanced: benign = 56%, malignant = 27%, normal = 17%.
This causes models to be biased toward the majority class (benign).
Two complementary strategies are used:

### Strategy 1: Weighted Random Sampler (oversampling rare classes during training)
```python
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets

train_dataset = datasets.ImageFolder("data/processed/train", transform=train_transform)
val_dataset   = datasets.ImageFolder("data/processed/val",   transform=val_test_transform)
test_dataset  = datasets.ImageFolder("data/processed/test",  transform=val_test_transform)

# Count samples per class in training set
class_counts = [0] * len(train_dataset.classes)
for _, label in train_dataset.samples:
    class_counts[label] += 1

# Weight: rare classes get higher probability of being sampled
class_weights = [1.0 / c for c in class_counts]
sample_weights = [class_weights[label] for _, label in train_dataset.samples]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)
```

### Strategy 2: Weighted Cross-Entropy Loss
```python
# Give higher loss penalty for misclassifying rare classes
weights = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=weights)
```

> **Report justification:** Both strategies reduce the model's tendency to predict the
> majority class. The sampler ensures balanced mini-batches during training; the weighted
> loss penalizes errors on rare classes more heavily. This is especially important for
> malignant detection: missing a malignant case is a serious clinical error (false negative).

---

## 9. Model Architecture

### Chosen Model: ResNet-18 with Transfer Learning

```python
import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=3, freeze_backbone=True):
    """
    ResNet-18 pretrained on ImageNet, with replaced classification head.

    Args:
        num_classes: 3 (benign, malignant, normal)
        freeze_backbone: if True, only train the final FC layer (Stage 1)
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        # Freeze all layers — only the new FC head will be trained
        for param in model.parameters():
            param.requires_grad = False

    # Replace the original 1000-class ImageNet head with a 3-class head
    in_features = model.fc.in_features   # 512 for ResNet-18
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),               # Regularization to reduce overfitting
        nn.Linear(in_features, num_classes)
    )

    return model
```

### Why ResNet-18?

| Criterion | Justification |
|-----------|--------------|
| Transfer learning | ImageNet features (edges, textures, shapes) transfer well to medical images |
| Small dataset | 546 training images is small; pretrained weights provide a strong starting point |
| Speed | ResNet-18 (~11M params) trains in minutes even on CPU |
| Literature support | Well-validated in medical imaging classification tasks |
| Two-stage training | Freeze backbone first → prevents overfitting early; unfreeze later for fine-tuning |

### Training in Two Stages

**Stage 1 — Warm-up (backbone frozen, only train new head):**
- Epochs: 10
- Learning rate: 1e-3
- Purpose: The new FC layer has random weights — letting the whole network
  train immediately would destroy the pretrained features

**Stage 2 — Fine-tuning (unfreeze all layers):**
- Epochs: 20 more
- Learning rate: 1e-4 (much lower — we only want small adjustments to pretrained weights)
- Purpose: Adapt all layers to the ultrasound domain

---

## 10. Hyperparameter Optimization

**The validation set is used exclusively for all hyperparameter decisions.**
Never use the test set to select hyperparameters.

### Hyperparameters to tune (one at a time, or use a small grid)

| Hyperparameter | Values to try | Selection criterion |
|----------------|---------------|---------------------|
| Learning rate (Stage 1) | 1e-2, **1e-3**, 1e-4 | Val loss after 10 epochs |
| Learning rate (Stage 2) | 1e-3, **1e-4**, 1e-5 | Val loss after fine-tuning |
| Batch size | 16, **32** | Val accuracy |
| Dropout rate in head | 0.3, **0.4**, 0.5 | Val loss |
| Epochs Stage 1 | 5, **10**, 15 | Val loss plateau |

> Bold = recommended starting values based on literature for similar tasks.
> For a 3-page report, you do NOT need an exhaustive grid search — try 3–5
> combinations and report the validation results in a small table.

### Minimal hyperparameter experiment table (include in report)

| Experiment | LR Stage1 | LR Stage2 | Dropout | Val Loss | Val Acc | Val AUROC |
|-----------|-----------|-----------|---------|----------|---------|-----------|
| Exp 1     | 1e-2      | 1e-3      | 0.3     | ...      | ...     | ...       |
| Exp 2     | **1e-3**  | **1e-4**  | **0.4** | ...      | ...     | ...       |
| Exp 3     | 1e-4      | 1e-5      | 0.5     | ...      | ...     | ...       |

**Best configuration** = lowest val loss / highest val AUROC → use this for final test evaluation.

### Early Stopping (prevents overfitting, monitored on validation loss)
```python
class EarlyStopping:
    """
    Stops training if val_loss does not improve for `patience` epochs.
    Saves the best model checkpoint automatically.
    """
    def __init__(self, patience=7, min_delta=0.001, checkpoint_path="outputs/models/best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            print(f"  ✓ Val loss improved → checkpoint saved")
        else:
            self.counter += 1
            print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print("  Early stopping triggered.")
```

### Learning Rate Scheduler
```python
# Reduce LR by half if val_loss doesn't improve for 3 epochs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3, verbose=True
)
# Call after each epoch: scheduler.step(val_loss)
```

---

## 11. Training Loop

```python
import torch
import pandas as pd
from tqdm import tqdm

def run_epoch(model, loader, optimizer, criterion, device, is_train=True):
    """Run one epoch. Returns (avg_loss, accuracy)."""
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for X, y in tqdm(loader, leave=False):
            X, y = X.to(device), y.to(device)
            if is_train:
                optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * X.size(0)
            correct    += (out.argmax(1) == y).sum().item()
            total      += X.size(0)

    return total_loss / total, correct / total


def train_model(model, train_loader, val_loader, optimizer, criterion,
                scheduler, early_stopping, device, num_epochs=30):
    """Full training loop with validation monitoring."""
    log = []

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer,
                                          criterion, device, is_train=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   optimizer,
                                          criterion, device, is_train=False)

        scheduler.step(val_loss)
        early_stopping(val_loss, model)

        log.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc,  4),
            "val_loss":   round(val_loss,   4),
            "val_acc":    round(val_acc,    4)
        })

        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.3f}")

        if early_stopping.early_stop:
            print(f"Stopped at epoch {epoch}.")
            break

    pd.DataFrame(log).to_csv("outputs/logs/training_log.csv", index=False)
    return log


# ─── Full pipeline ───────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Stage 1: Train head only (backbone frozen)
model = get_model(num_classes=3, freeze_backbone=True).to(device)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
early_stopping = EarlyStopping(patience=7)

print("=== Stage 1: Training classification head (backbone frozen) ===")
train_model(model, train_loader, val_loader, optimizer, criterion,
            scheduler, early_stopping, device, num_epochs=15)

# Stage 2: Fine-tune all layers
print("\n=== Stage 2: Fine-tuning full network ===")
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
early_stopping = EarlyStopping(patience=7, checkpoint_path="outputs/models/best_model_finetuned.pth")

train_model(model, train_loader, val_loader, optimizer, criterion,
            scheduler, early_stopping, device, num_epochs=25)

# Load best checkpoint for evaluation
model.load_state_dict(torch.load("outputs/models/best_model_finetuned.pth"))
print("Best checkpoint loaded.")
```

---

## 12. Evaluation & Performance Metrics

### Why each metric matters clinically

| Metric | Formula | Clinical importance |
|--------|---------|-------------------|
| **Sensitivity (Recall)** | TP / (TP + FN) | Most critical — missing malignant = missed cancer |
| **Specificity** | TN / (TN + FP) | Avoiding unnecessary biopsies from false positives |
| **Precision (PPV)** | TP / (TP + FP) | Of all malignant predictions, how many are correct |
| **F1-score** | 2 × (P×R)/(P+R) | Balances precision and recall; good for imbalanced classes |
| **AUROC** | Area under ROC | Overall discriminative ability independent of threshold |
| **Accuracy** | (TP+TN)/Total | Overall, but misleading with class imbalance |

> **Report minimum requirement:** sensitivity + specificity OR precision + recall, PLUS at least one more.
> **Recommended:** report all of the above, per-class.

### Full evaluation code
```python
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, RocCurveDisplay, roc_curve
)
import matplotlib.pyplot as plt

CLASS_NAMES = train_dataset.classes   # ['benign', 'malignant', 'normal']

def evaluate_model(model, loader, device, class_names, split_name="Test"):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    # ── 1. Classification Report (precision, recall, F1 per class) ──────────
    print(f"\n{'='*60}")
    print(f"  {split_name} Set Classification Report")
    print('='*60)
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # ── 2. Per-class Sensitivity & Specificity ──────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    print("\nPer-class Sensitivity & Specificity:")
    for i, cls in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp           # false negatives for this class
        fp = cm[:, i].sum() - tp           # false positives for this class
        tn = cm.sum() - tp - fn - fp       # true negatives
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"  {cls:12s}  Sensitivity: {sens:.3f}   Specificity: {spec:.3f}")

    # ── 3. AUROC (One-vs-Rest, macro average) ───────────────────────────────
    auroc_macro = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    print(f"\nMacro-average AUROC (OvR): {auroc_macro:.4f}")

    # ── 4. Confusion Matrix Figure ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {split_name} Set")
    plt.tight_layout()
    plt.savefig(f"outputs/figures/fig4_confusion_matrix_{split_name.lower()}.png", dpi=150)
    plt.show()

    # ── 5. ROC Curves (per class) ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve((all_labels == i).astype(int), all_probs[:, i])
        auc_score = roc_auc_score((all_labels == i).astype(int), all_probs[:, i])
        ax.plot(fpr, tpr, label=f"{cls} (AUROC = {auc_score:.3f})")
    ax.plot([0,1],[0,1],'k--', label="Random")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(f"ROC Curves per Class — {split_name} Set")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/figures/fig5_roc_curves_{split_name.lower()}.png", dpi=150)
    plt.show()

    return {"auroc": auroc_macro, "cm": cm, "preds": all_preds,
            "labels": all_labels, "probs": all_probs}


# Evaluate on VALIDATION set (to verify tuning is working)
val_results  = evaluate_model(model, val_loader,  device, CLASS_NAMES, "Validation")

# Evaluate on TEST set — do this ONLY ONCE at the very end
test_results = evaluate_model(model, test_loader, device, CLASS_NAMES, "Test")
```

### Training Curves Figure
```python
import pandas as pd
import matplotlib.pyplot as plt

log = pd.read_csv("outputs/logs/training_log.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(log["epoch"], log["train_loss"], label="Train Loss")
ax1.plot(log["epoch"], log["val_loss"],   label="Val Loss")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.set_title("Training and Validation Loss"); ax1.legend()

ax2.plot(log["epoch"], log["train_acc"], label="Train Acc")
ax2.plot(log["epoch"], log["val_acc"],   label="Val Acc")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
ax2.set_title("Training and Validation Accuracy"); ax2.legend()

plt.tight_layout()
plt.savefig("outputs/figures/fig3_training_curves.png", dpi=150)
plt.show()
```

---

## 13. Explainability — Grad-CAM

Grad-CAM highlights which regions of the image the model focuses on.
This is clinically relevant: a trustworthy model should activate over the lesion area.

```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from PIL import Image

# Target the last convolutional layer in ResNet-18
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

def visualize_gradcam(img_path, true_label, model, cam, transform, class_names, device):
    img_pil = Image.open(img_path).convert("RGB")
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        logits = model(input_tensor)
        pred_class = logits.argmax(1).item()
        pred_prob  = F.softmax(logits, dim=1)[0, pred_class].item()

    # Generate Grad-CAM for predicted class
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=[ClassifierOutputTarget(pred_class)])[0]

    # Overlay on image
    img_np = np.array(img_pil.resize((224, 224))) / 255.0
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img_pil.resize((224, 224)), cmap="gray")
    axes[0].set_title(f"Original\nTrue: {class_names[true_label]}")
    axes[0].axis("off")
    axes[1].imshow(visualization)
    axes[1].set_title(f"Grad-CAM\nPred: {class_names[pred_class]} ({pred_prob:.2f})")
    axes[1].axis("off")
    plt.tight_layout()
    return fig

# Show one example per class
for cls_idx, cls_name in enumerate(CLASS_NAMES):
    sample_path = list(Path(f"data/processed/test/{cls_name}").glob("*.png"))[0]
    fig = visualize_gradcam(sample_path, cls_idx, model, cam,
                             val_test_transform, CLASS_NAMES, device)
    fig.savefig(f"outputs/figures/gradcam_{cls_name}.png", dpi=150)
    plt.show()
```

---

## 14. Figures & Tables for Report

All figures go in `outputs/figures/`. They are NOT counted toward the 3-page limit.

| Figure | File | Content |
|--------|------|---------|
| **Fig 1** | `fig1_sample_images.png` | 3×2 grid of sample US images (1 per class + corresponding mask) |
| **Fig 2** | `fig2_class_distribution.png` | Bar chart of class counts in train / val / test splits |
| **Fig 3** | `fig3_training_curves.png` | Loss and accuracy curves over epochs (train vs val) |
| **Fig 4** | `fig4_confusion_matrix_test.png` | Confusion matrix on test set |
| **Fig 5** | `fig5_roc_curves_test.png` | Per-class ROC curves with AUROC values |

### Sample images figure code
```python
fig, axes = plt.subplots(2, 3, figsize=(10, 7))
for col, cls in enumerate(CLASS_NAMES):
    imgs = list(Path(f"data/raw/{cls}").glob("*.png"))
    # Original image
    sample_img  = [f for f in imgs if "_mask" not in f.name][0]
    # Its corresponding mask
    sample_mask = Path(str(sample_img).replace(".png", "_mask.png"))

    axes[0, col].imshow(Image.open(sample_img),  cmap="gray")
    axes[0, col].set_title(f"{cls.capitalize()}")
    axes[0, col].axis("off")

    axes[1, col].imshow(Image.open(sample_mask), cmap="gray")
    axes[1, col].set_title(f"{cls.capitalize()} (mask)")
    axes[1, col].axis("off")

fig.suptitle("Figure 1: Sample Ultrasound Images and Ground-Truth Masks per Class",
             fontsize=13)
plt.tight_layout()
plt.savefig("outputs/figures/fig1_sample_images.png", dpi=150)
plt.show()
```

### Class distribution figure code
```python
splits = ["train", "val", "test"]
x = np.arange(len(CLASS_NAMES))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 5))
for i, split in enumerate(splits):
    counts = [len(list(Path(f"data/processed/{split}/{cls}").glob("*.png")))
              for cls in CLASS_NAMES]
    ax.bar(x + i * width, counts, width, label=split.capitalize())

ax.set_xticks(x + width)
ax.set_xticklabels([c.capitalize() for c in CLASS_NAMES])
ax.set_ylabel("Number of Images")
ax.set_title("Figure 2: Class Distribution Across Train / Validation / Test Splits")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/figures/fig2_class_distribution.png", dpi=150)
plt.show()
```

---

## 15. Report Outline (3 pages)

### Page 1 — Introduction + Methods (beginning)

**Introduction (~0.3 page)**
- Breast cancer epidemiology: global incidence and mortality (cite WHO, 2024)
- Role of ultrasound in breast cancer screening: advantages over mammography for
  dense breast tissue, no radiation, low cost
- Challenge: inter-reader variability; radiologist shortage in low-resource settings
- Objective: train a CNN to classify breast US images as normal / benign / malignant
  as a clinical decision support tool

**Methods (~0.8 page)**

*Dataset:*
- Al-Dhabyani et al. 2020: 780 images, 600 patients, 3 classes
- Mask files excluded from classification inputs

*Data splits:*
- 70% train / 15% validation / 15% test (stratified by class, random seed = 42)
- Validation set role: hyperparameter tuning, model selection, early stopping
- Test set role: final unbiased performance evaluation (used once)

*Preprocessing:*
- Resize to 224×224; grayscale→3-channel; ImageNet normalization
- Augmentation (train only): horizontal flip, ±15° rotation, brightness/contrast jitter

*Class imbalance handling:*
- WeightedRandomSampler + class-weighted cross-entropy loss

*Model:*
- ResNet-18, pretrained on ImageNet; FC head replaced (512→3 with Dropout 0.4)
- Two-stage training: Stage 1 frozen backbone (lr=1e-3, 15 epochs);
  Stage 2 full fine-tuning (lr=1e-4, up to 25 epochs)
- Early stopping (patience=7) on validation loss

*Hyperparameter optimization:*
- Learning rate, dropout, batch size tuned using validation AUROC
- Table of experiments with val loss and val AUROC (see Table 1)

*Metrics:*
- Per-class: sensitivity, specificity, precision, F1-score
- Overall: macro AUROC, accuracy

### Page 2 — Methods (end) + Results

*Performance metrics justification (~0.1 page):*
- Sensitivity for malignant class is the primary clinical metric (cost of FN >> FP)
- AUROC is threshold-independent, suitable for comparing across experiments
- F1 accounts for class imbalance better than accuracy

**Results (~0.5 page)**
- Table 2: Per-class sensitivity, specificity, precision, F1 (test set)
- Macro AUROC on test set (e.g., "0.87")
- Figure 4 (confusion matrix): reference here — note which classes are most confused
- Figure 5 (ROC curves): reference here — note which class has best/worst AUROC
- Training curves (Figure 3): when early stopping triggered; no overfitting evidence

### Page 3 — Discussion + References

**Discussion (~0.6 page)**
- Primary finding: model achieves [X] AUROC; malignant sensitivity = [Y]
- Benign vs. malignant confusion: clinically important, ultrasound features overlap
- Normal class: often well-classified (most distinct appearance)
- Grad-CAM (if included): activations spatially consistent with lesion locations —
  or not — and what that implies for trustworthiness
- **Limitations:**
  - Small single-institution dataset (780 images, 1 hospital)
  - No external validation cohort
  - No patient-level metadata (age, symptoms) used
  - Grayscale-to-RGB conversion is an approximation
  - Masks available but not used in classification (segmentation-guided approach unexplored)
- **Recommended next steps:**
  - Larger multi-center dataset for external validation
  - Incorporate segmentation masks to guide attention
  - Test on prospective clinical data
  - Explore uncertainty quantification (conformal prediction, MC Dropout)

**References (not counted in 3 pages)**
See Section 21 for full reference list.

---

## 16. Flash Talk Slide Structure

**Total: 2 minutes = ~120 seconds. Suggested 7 slides.**

| Slide | Title | Content | Time |
|-------|-------|---------|------|
| 1 | Title | Project title, your name, course, date | 8 sec |
| 2 | Clinical Problem | "Why does this matter?" — breast cancer stats, US role, AI motivation. 2–3 bullet points max | 18 sec |
| 3 | Dataset & Task | Show Fig 1 (sample images, 3 classes). State: 780 images, 3 classes, your split (70/15/15) | 18 sec |
| 4 | Methods | Architecture diagram or bullet list: ResNet-18, 2-stage training, augmentation, imbalance handling | 22 sec |
| 5 | Validation & Tuning | Brief note: how you used the val set to pick hyperparameters (Table 1 or 1-2 sentences) | 12 sec |
| 6 | Results | Show Fig 4 (confusion matrix) or Table 2 (sensitivity/specificity/F1/AUROC) | 14 sec |
| 7 | Limitations & Conclusion | 2–3 bullet limitations; 1 takeaway sentence. End with "Thank you / Questions?" | 8 sec |

> 💡 **Recording tip:** Practice out loud 3 times before recording. Use a timer.
> Record in a quiet room. Use screen recording + webcam optional.
> Export Powerpoint as PDF for submission.

---

## 17. AI Use Documentation Template

**File: `AI_use_documentation.md`** (required; not counted in page limit)

```markdown
# AI Use Documentation — M7016H Final Project
Student: [Your Name]
Date submitted: [Date]

In accordance with Luleå tekniska universitets AI policy, all uses of AI tools
in this project are documented below. AI was used as a supplementary tool only.
No AI-generated text appears verbatim in the report or reflection.

| # | Date       | Tool    | Exact Prompt Used                                      | Output Received         | How I Used / Modified It                         |
|---|------------|---------|--------------------------------------------------------|-------------------------|--------------------------------------------------|
| 1 | YYYY-MM-DD | Claude  | "Explain what AUROC means in multiclass classification"| Conceptual explanation  | Paraphrased into Methods section; rewrote fully  |
| 2 | YYYY-MM-DD | Claude  | "What is a WeightedRandomSampler in PyTorch and why use it for imbalanced data?" | Code + explanation | Used code as starting point; adapted and tested  |
| 3 | YYYY-MM-DD | ChatGPT | "..."                                                  | ...                     | ...                                              |

## Statement
All text in the report and reflection is my own writing. AI tools were used only
for conceptual explanations and code starting points, all of which were verified,
tested, and adapted by me. Critical analysis, interpretation of results, and
clinical reasoning are entirely my own.
```

---

## 18. Day-by-Day Schedule

**Today: April 21, 2026 | Deadline: May 15, 2026 = 24 days**

### 📅 WEEK 1: Setup, EDA, Data Preparation (Apr 21–27)

| Day | Date | Tasks | Deliverable |
|-----|------|-------|------------|
| 1 | Apr 21 | Download dataset from Dropbox. Set up Python environment (local or Colab). Install requirements. Create folder structure. | Environment ready |
| 2 | Apr 22 | `01_EDA.ipynb`: Count images per class, visualize samples from all 3 classes, verify mask files exist, note class imbalance. Plot Fig 2 draft. | EDA notebook done |
| 3 | Apr 23 | `02_Preprocessing.ipynb`: Write and run 70/15/15 stratified split with seed=42. Filter out mask files. Verify counts. | data/processed/ populated |
| 4 | Apr 24 | Implement `val_test_transform` and `train_transform`. Create PyTorch Datasets for all 3 splits. Verify DataLoaders return correct shapes. | DataLoaders working |
| 5 | Apr 25 | Implement WeightedRandomSampler + weighted CE loss. Visualize augmented training samples to sanity-check. Generate Fig 1 (sample images). | Augmentation verified |
| 6 | Apr 26 | Set up ResNet-18 model. Test forward pass with one batch. Implement EarlyStopping and LR scheduler. | Model architecture ready |
| 7 | Apr 27 | Run a 2-epoch smoke test of the full pipeline (Stage 1). Check that loss decreases and no errors occur. Fix any bugs. | Pipeline smoke test passes |

> 💡 **Office Hours: Apr 28, 9–11 CET** — attend with questions about preprocessing or architecture!

---

### 📅 WEEK 2: Training, Hyperparameter Tuning, Evaluation (Apr 28 – May 4)

| Day | Date | Tasks | Deliverable |
|-----|------|-------|------------|
| 8 | Apr 28 | `03_Model_Training.ipynb`: Run Stage 1 (frozen backbone, 15 epochs). Log train/val loss and accuracy. Save checkpoint. | Stage 1 training log |
| 9 | Apr 29 | Run Stage 2 (unfreeze all, lr=1e-4, up to 25 epochs). Monitor val loss. Early stopping should save best checkpoint. | Best model checkpoint saved |
| 10 | Apr 30 | `04_Hyperparameter_Tuning.ipynb`: Try 2–3 variants (different lr, dropout). Record val loss + val AUROC for each. Pick best config. | Hyperparameter table (Table 1) |
| 11 | May 1 | `05_Evaluation.ipynb`: Load best checkpoint. Run `evaluate_model()` on VALIDATION set to confirm results. | Val metrics confirmed |
| 12 | May 2 | Run `evaluate_model()` on TEST set — **only once!** Generate all 5 figures. Save all to `outputs/figures/`. | All figures generated |
| 13 | May 3 | `06_GradCAM_Explainability.ipynb`: Generate Grad-CAM for 1 example per class. Assess if activations are over lesion. | Grad-CAM images |
| 14 | May 4 | Code cleanup: add comments, docstrings, remove debug prints. Confirm all notebooks run top-to-bottom without errors. | Clean, runnable code |

> 💡 **Office Hours: May 4, 9–11 CET** — ideal timing: results are done, writing begins!

---

### 📅 WEEK 3: Report Writing + Presentation (May 5–11)

| Day | Date | Tasks | Deliverable |
|-----|------|-------|------------|
| 15 | May 5 | Draft **Introduction** (0.3 page). Find and add 3–4 references. Write clinical motivation clearly. | Introduction draft |
| 16 | May 6 | Draft **Methods — Dataset, splits, preprocessing** subsections. Justify every choice. | Methods Part 1 draft |
| 17 | May 7 | Draft **Methods — Model, training, hyperparameter tuning** subsections. Insert Table 1. | Methods Part 2 draft |
| 18 | May 8 | Draft **Results** section. Insert Table 2 (metrics), reference Figures 3–5. | Results draft |
| 19 | May 9 | Draft **Discussion**: findings interpretation, limitations, next steps. Write **References** list. | Discussion + References draft |
| 20 | May 10 | Complete `AI_use_documentation.md`. Check page count (exactly 3 pages for main text; figures/tables separate). Revise and tighten. | Complete report draft |
| 21 | May 11 | Create 7 slides in PowerPoint. Add figures. Time yourself: must be ≤2 min. Record flash talk (re-record if needed). Export slides as PDF. | Video + slides.pdf ready |

---

### 📅 WEEK 4: Final Polish & Submission (May 12–15)

| Day | Date | Tasks | Deliverable |
|-----|------|-------|------------|
| 22 | May 12 | Final proofread: citations match references, all figures numbered and captioned, page limit respected. | Polished report |
| 23 | May 13 | Final code review: no data files included, no hard-coded absolute paths, notebooks run cleanly. | Submission-ready code |
| 24 | May 14 | Export report as PDF. Compress video if >100 MB. Do final review of all 4 submission items. | 4 files ready |
| 25 | May 15 | ✅ **SUBMIT on Canvas by 23:59:** (1) report.pdf, (2) video, (3) slides.pdf, (4) code. Screenshot confirmation. | **SUBMITTED** |

---

### 📅 Symposium & Reflection (May 16–26)

| Day | Date | Tasks |
|-----|------|-------|
| May 21 | Symposium | Watch ALL peer presentations. Take structured notes: dataset used, model, key metric, clinical task, one strength, one limitation per project. |
| May 22–25 | — | Write 1-page reflection: assess your own project's clinical impact; assess 2–3 peers' projects critically; connect to course learning goals. |
| May 26 | — | ✅ **SUBMIT reflection.pdf on Canvas by 23:59.** |

---

## 19. Submission Checklist

### Due May 15, 2026 @ 23:59

- [ ] **report.pdf** — exactly 3 pages main text; figures/tables/AI docs as appendix
  - [ ] Introduction with clinical justification
  - [ ] Methods: split (70/15/15), preprocessing, augmentation, imbalance handling,
    model, hyperparameter tuning with validation set, metrics justification
  - [ ] Results: sensitivity, specificity + at least one of F1, AUROC, accuracy
  - [ ] Discussion: findings, limitations, next steps
  - [ ] Reference list (at least 4 citations)
  - [ ] Up to 5 figures/tables (not in page count)
  - [ ] AI use documentation (not in page count)

- [ ] **Video (flash talk, exactly ≤2 min)** — covers intro, methods, results, conclusion

- [ ] **slides.pdf** — exported from PowerPoint or equivalent

- [ ] **Code** — all notebooks/scripts; no data files; runs without errors

### Due May 26, 2026 @ 23:59

- [ ] **reflection.pdf** — 1 page; discusses your project AND peers' work;
  references the symposium (May 21); in PDF format; no AI-generated text

---

## 20. Common Pitfalls

| Pitfall | How to avoid it |
|---------|----------------|
| ❌ Mask images in dataset | Filter out `_mask.png` files explicitly in split script |
| ❌ Missing validation set | Use 70/15/15 split; tune on val, report on test |
| ❌ Test set used for tuning | Touch test set **only once**, after all decisions are made |
| ❌ Reporting only accuracy | Add sensitivity/specificity + AUROC or F1; accuracy is misleading |
| ❌ No random seed | Set `random.seed(42)` AND `torch.manual_seed(42)` for reproducibility |
| ❌ Augmenting val/test | Apply augmentation ONLY to training data |
| ❌ AI-generated report text | Write all report text yourself; document AI use in appendix |
| ❌ No justification for choices | Every methodological choice must have a 1-sentence rationale in the report |
| ❌ Flash talk > 2 min | Practice 3× with a timer; cut ruthlessly |
| ❌ Not attending symposium | Required to write the May 26 reflection |

---

## 21. Key References

1. **Dataset:** Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound
   images. *Data in Brief.* 2020 Feb;28:104863. DOI: 10.1016/j.dib.2019.104863

2. **ResNet:** He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition.
   *Proceedings of the IEEE CVPR.* 2016:770-778.

3. **Transfer learning in medical imaging:** Tajbakhsh N, et al. Convolutional neural
   networks for medical image analysis: Full training or fine tuning? *IEEE Trans Med
   Imaging.* 2016;35(5):1299-1312.

4. **Breast US AI review:** Shen L, et al. Deep learning to improve breast cancer
   detection on screening mammography. *Sci Rep.* 2019;9:12495.

5. **Clinical motivation:** World Health Organization. Breast cancer fact sheet. 2024.
   https://www.who.int/news-room/fact-sheets/detail/breast-cancer

6. **Grad-CAM:** Selvaraju RR, et al. Grad-CAM: Visual explanations from deep networks
   via gradient-based localization. *Proceedings of the IEEE ICCV.* 2017:618-626.

---

*Good luck! The most important habit: commit your code and notes daily, and
use the Office Hours on Apr 28 and May 4 — they are your best checkpoints.*
