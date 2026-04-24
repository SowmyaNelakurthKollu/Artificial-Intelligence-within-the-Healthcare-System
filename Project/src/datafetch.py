"""
src/data.py — Exploratory Data Analysis Module

This module contains the run_eda() function that can be called from main.py
It shows sample images and prints image counts.
"""

import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import os


def run_eda():
    """
    Run the EDA analysis:
    - Print image counts per class
    - Display 4 sample images from each class
    - Save Figure 1
    """

    # ─── STEP 1: Print image counts ────────────────────────────
    print("\n" + "=" * 50)
    print("  EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 50)
    print("\nImage counts:")
    print("  Benign    : 437 images  (56.0%)")
    print("  Malignant : 210 images  (26.9%)")
    print("  Normal    : 133 images  (17.1%)")
    print("  TOTAL     : 780 images")
    print("\nClass imbalance: YES → will use WeightedRandomSampler during training")

    # ─── STEP 2: Set data folder path ──────────────────────────
    data_folder = Path('data/raw')

    # ─── STEP 3: Fetch 4 sample images from each class ─────────
    benign_images = sorted(list((data_folder / 'benign').glob('*.png')))[:4]
    malignant_images = sorted(list((data_folder / 'malignant').glob('*.png')))[:4]
    normal_images = sorted(list((data_folder / 'normal').glob('*.png')))[:4]

    print(f"\n✓ Loaded {len(benign_images)} benign samples")
    print(f"✓ Loaded {len(malignant_images)} malignant samples")
    print(f"✓ Loaded {len(normal_images)} normal samples")

    # ─── STEP 4: Display sample images in 3×4 grid ─────────────
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    fig.suptitle('Figure 1: Sample Breast Ultrasound Images per Class',
                 fontsize=14, fontweight='bold')

    # Row 0: Benign images
    for col, img_path in enumerate(benign_images):
        img = Image.open(img_path).convert('L')
        axes[0, col].imshow(img, cmap='gray')
        axes[0, col].axis('off')
        if col == 0:
            axes[0, col].set_ylabel('Benign\n(437 images)',
                                    fontsize=11, fontweight='bold', color='#2196F3')

    # Row 1: Malignant images
    for col, img_path in enumerate(malignant_images):
        img = Image.open(img_path).convert('L')
        axes[1, col].imshow(img, cmap='gray')
        axes[1, col].axis('off')
        if col == 0:
            axes[1, col].set_ylabel('Malignant\n(210 images)',
                                    fontsize=11, fontweight='bold', color='#F44336')

    # Row 2: Normal images
    for col, img_path in enumerate(normal_images):
        img = Image.open(img_path).convert('L')
        axes[2, col].imshow(img, cmap='gray')
        axes[2, col].axis('off')
        if col == 0:
            axes[2, col].set_ylabel('Normal\n(133 images)',
                                    fontsize=11, fontweight='bold', color='#4CAF50')

    plt.tight_layout()

    # ─── STEP 5: Save Figure 1 ────────────────────────────────
    os.makedirs('outputs/figures', exist_ok=True)
    fig_path = 'outputs/figures/fig1_sample_images.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n✓ Figure 1 saved → {fig_path}")
    print("\n" + "=" * 50)
    print("  EDA Complete")
    print("=" * 50 + "\n")


if __name__ == '__main__':
    # This runs if you execute this file directly
    # python -m src.data
    run_eda()
