"""
src/dataevaluation.py — Model Evaluation Module

This module evaluates the trained model on test set and generates:
- Confusion matrix (Figure 4)
- ROC curves (Figure 5)
- Classification metrics (Table 2)
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pathlib import Path
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    precision_recall_fscore_support
)


def build_model():
    """Rebuild the trained model architecture."""
    model = models.resnet18(weights=None)
    in_feats = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_feats, 3)
    )
    return model


def get_predictions(model, loader, device):
    """Get predictions on a dataset."""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def evaluate_set(preds, labels, probs, class_names, split_name='Test'):
    """Evaluate and print metrics."""

    print('=' * 60)
    print(f'  RESULTS ON {split_name.upper()} SET')
    print('=' * 60)

    # Classification report
    print('\nClassification Report:\n')
    print(classification_report(labels, preds, target_names=class_names))

    # Per-class sensitivity and specificity
    cm = confusion_matrix(labels, preds)

    print('Per-class Sensitivity and Specificity:')
    print(f'{"Class":<12}  {"Sensitivity":>13}  {"Specificity":>13}')
    print('-' * 42)

    for i, cls in enumerate(class_names):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - TP - FN - FP

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        print(f'{cls:<12}  {sensitivity:>13.3f}  {specificity:>13.3f}')

    # Accuracy and AUROC
    accuracy = (preds == labels).mean()
    auroc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')

    print(f'\nOverall Accuracy : {accuracy:.4f}')
    print(f'Macro AUROC      : {auroc:.4f}')
    print()

    # Figure 4: Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=[c.capitalize() for c in class_names])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'Figure 4: Confusion Matrix — {split_name} Set', fontsize=12, fontweight='bold')

    plt.tight_layout()
    os.makedirs('outputs/figures', exist_ok=True)
    fig4_path = f'outputs/figures/fig4_confusion_matrix_{split_name.lower()}.png'
    plt.savefig(fig4_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Figure 4 saved → {fig4_path}')

    # Figure 5: ROC Curves
    colors = ['#2196F3', '#F44336', '#4CAF50']

    fig, ax = plt.subplots(figsize=(7, 6))

    for i, (cls, color) in enumerate(zip(class_names, colors)):
        binary_labels = (labels == i).astype(int)
        class_probs = probs[:, i]

        fpr, tpr, _ = roc_curve(binary_labels, class_probs)
        auc = roc_auc_score(binary_labels, class_probs)

        ax.plot(fpr, tpr, label=f'{cls.capitalize()}  (AUROC = {auc:.3f})',
                color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUROC = 0.5)')

    ax.set_xlabel('False Positive Rate  (1 - Specificity)', fontsize=11)
    ax.set_ylabel('True Positive Rate  (Sensitivity)', fontsize=11)
    ax.set_title(f'Figure 5: ROC Curves per Class — {split_name} Set', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig5_path = f'outputs/figures/fig5_roc_curves_{split_name.lower()}.png'
    plt.savefig(fig5_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Figure 5 saved → {fig5_path}')

    return {'accuracy': accuracy, 'auroc': auroc, 'cm': cm}


def run_evaluation():
    """Evaluate model on validation and test sets."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load datasets
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    val_dataset = datasets.ImageFolder('data/processed/val', transform=eval_transform)
    test_dataset = datasets.ImageFolder('data/processed/test', transform=eval_transform)

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    CLASS_NAMES = val_dataset.classes

    print(f"Val dataset  : {len(val_dataset)} images")
    print(f"Test dataset : {len(test_dataset)} images")

    # Load model
    model = build_model().to(device)
    model.load_state_dict(torch.load('outputs/models/best_model_stage2.pth', map_location=device))

    print("\n" + "=" * 60)
    print("  EVALUATION")
    print("=" * 60)

    # Evaluate on validation set
    print("\nRunning predictions on validation set...")
    val_preds, val_labels, val_probs = get_predictions(model, val_loader, device)
    val_results = evaluate_set(val_preds, val_labels, val_probs, CLASS_NAMES, 'Validation')

    # Evaluate on test set
    print("\nRunning predictions on TEST set...")
    print("(This is the final result — run only once!)")
    test_preds, test_labels, test_probs = get_predictions(model, test_loader, device)
    test_results = evaluate_set(test_preds, test_labels, test_probs, CLASS_NAMES, 'Test')

    # Print Table 2
    print("\n" + "=" * 80)
    print("  TABLE 2: Test Set Performance Metrics")
    print("=" * 80)

    precision, recall, f1, support = precision_recall_fscore_support(
        test_labels, test_preds, labels=[0, 1, 2]
    )

    per_class_auroc = []
    for i in range(len(CLASS_NAMES)):
        binary = (test_labels == i).astype(int)
        auc = roc_auc_score(binary, test_probs[:, i])
        per_class_auroc.append(auc)

    cm = confusion_matrix(test_labels, test_preds)
    sensitivities = []
    specificities = []

    for i in range(len(CLASS_NAMES)):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - TP - FN - FP
        sensitivities.append(TP / (TP + FN) if (TP + FN) > 0 else 0)
        specificities.append(TN / (TN + FP) if (TN + FP) > 0 else 0)

    print(f'{"Class":<12}  {"Sensitivity":>12}  {"Specificity":>12}  '
          f'{"Precision":>10}  {"F1":>8}  {"AUROC":>8}  {"N":>5}')
    print('-' * 80)

    for i, cls in enumerate(CLASS_NAMES):
        print(f'{cls:<12}  {sensitivities[i]:>12.3f}  {specificities[i]:>12.3f}  '
              f'{precision[i]:>10.3f}  {f1[i]:>8.3f}  {per_class_auroc[i]:>8.3f}  {support[i]:>5}')

    print('-' * 80)

    overall_acc = (test_preds == test_labels).mean()
    overall_auroc = roc_auc_score(test_labels, test_probs, multi_class='ovr', average='macro')

    print(f'\nOverall Accuracy : {overall_acc:.3f}  ({overall_acc*100:.1f}%)')
    print(f'Macro AUROC      : {overall_auroc:.3f}')
    print("=" * 80)

    print("\n✓ Evaluation complete!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    run_evaluation()
