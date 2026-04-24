"""
src/datatraining.py — Model Training Module

This module trains ResNet-18 in 2 stages:
Stage 1: Backbone frozen, train head only
Stage 2: Unfreeze all, fine-tune everything
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler


def build_model(freeze_backbone=True):
    """
    Build ResNet-18 with 3-class output head.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 3)
    )

    return model


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc='  Training', leave=False):
        images = images.to(device)
        labels = labels.to(device)

        predictions = model(images)
        loss = criterion(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (predictions.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    """Run validation."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='  Validating', leave=False):
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)
            loss = criterion(predictions, labels)

            total_loss += loss.item() * images.size(0)
            correct += (predictions.argmax(1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


class EarlyStopping:
    """Stop training if validation loss does not improve."""

    def __init__(self, patience=7, save_path='outputs/models/best_model.pth'):
        self.patience = patience
        self.save_path = save_path
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def check(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            torch.save(model.state_dict(), self.save_path)
            print(f'    Val loss improved → saved to {self.save_path}')
        else:
            self.counter += 1
            print(f'    No improvement ({self.counter}/{self.patience})')
            if self.counter >= self.patience:
                self.should_stop = True
                print('    Early stopping triggered!')


def run_training():
    """
    Train ResNet-18 in 2 stages.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # ─── Define transforms ────────────────────────────────────
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    val_test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    # ─── Load datasets ────────────────────────────────────────
    train_dataset = datasets.ImageFolder('data/processed/train', transform=train_transform)
    val_dataset = datasets.ImageFolder('data/processed/val', transform=val_test_transform)

    print(f"\nTrain dataset: {len(train_dataset)} images")
    print(f"Val dataset  : {len(val_dataset)} images")

    # ─── WeightedRandomSampler for class imbalance ───────────
    class_counts = [0, 0, 0]
    for _, label in train_dataset.samples:
        class_counts[label] += 1

    class_weights = [1.0 / c for c in class_counts]
    sample_weights = [class_weights[label] for _, label in train_dataset.samples]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"Train loader : {len(train_loader)} batches")
    print(f"Val loader   : {len(val_loader)} batches")

    # ─── Build model ──────────────────────────────────────────
    torch.manual_seed(42)
    model = build_model(freeze_backbone=True).to(device)

    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=7, save_path='outputs/models/best_model_stage1.pth')

    log = []

    # ─── STAGE 1: Train head only ─────────────────────────────
    print("\n" + "=" * 65)
    print("  STAGE 1: Training classification head (backbone frozen)")
    print("=" * 65)

    for epoch in range(1, 16):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f'Epoch {epoch:2d}/15  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.3f}  |  '
              f'Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.3f}')

        early_stopping.check(val_loss, model)
        log.append({'stage': 1, 'epoch': epoch, 'train_loss': round(train_loss, 4),
                    'train_acc': round(train_acc, 4), 'val_loss': round(val_loss, 4),
                    'val_acc': round(val_acc, 4)})

        if early_stopping.should_stop:
            print(f'Stopped early at epoch {epoch}')
            break

    print("\nStage 1 complete!")

    # ─── STAGE 2: Fine-tune all layers ────────────────────────
    print("\n" + "=" * 65)
    print("  STAGE 2: Fine-tuning full network (all layers unfrozen)")
    print("=" * 65)

    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=7, save_path='outputs/models/best_model_stage2.pth')

    for epoch in range(1, 26):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f'Epoch {epoch:2d}/25  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.3f}  |  '
              f'Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.3f}')

        early_stopping.check(val_loss, model)
        log.append({'stage': 2, 'epoch': epoch, 'train_loss': round(train_loss, 4),
                    'train_acc': round(train_acc, 4), 'val_loss': round(val_loss, 4),
                    'val_acc': round(val_acc, 4)})

        if early_stopping.should_stop:
            print(f'Stopped early at epoch {epoch}')
            break

    print("\nStage 2 complete!")

    # ─── Save training log ────────────────────────────────────
    os.makedirs('outputs/logs', exist_ok=True)
    df = pd.DataFrame(log)
    df.to_csv('outputs/logs/training_log.csv', index=False)
    print(f"Training log saved → outputs/logs/training_log.csv")

    # ─── Plot training curves ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Figure 3: Training and Validation Curves', fontsize=13, fontweight='bold')

    epoch_nums = list(range(1, len(log) + 1))

    axes[0].plot(epoch_nums, df['train_loss'], label='Train Loss', color='steelblue')
    axes[0].plot(epoch_nums, df['val_loss'], label='Val Loss', color='tomato')
    axes[0].set_title('Loss over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epoch_nums, df['train_acc'], label='Train Accuracy', color='steelblue')
    axes[1].plot(epoch_nums, df['val_acc'], label='Val Accuracy', color='tomato')
    axes[1].set_title('Accuracy over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('outputs/figures', exist_ok=True)
    plt.savefig('outputs/figures/fig3_training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Figure 3 saved → outputs/figures/fig3_training_curves.png")

    # ─── Load best model ──────────────────────────────────────
    model.load_state_dict(torch.load('outputs/models/best_model_stage2.pth', map_location=device))
    print("\n✓ Best model loaded and ready for evaluation!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    run_training()
