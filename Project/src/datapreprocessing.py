"""
src/datapreprocessing.py — Data Preprocessing & Train/Val/Test Split

This module creates the train/val/test split from the raw dataset.
"""

import os
import shutil
import random
from pathlib import Path
from torchvision import datasets, transforms


def run_preprocessing():
    """
    Create train/val/test split:
    - 70% train (546 images)
    - 15% val   (117 images)
    - 15% test  (117 images)
    """

    # ─── STEP 1: Configure paths and ratios ────────────────────
    SRC_DIR = Path('D:/Master Programme/Semester Two/Artificial Intelligence within the Healthcare System/Artificial-Intelligence-within-the-Healthcare-System/Project/data/raw')
    DEST_DIR = Path('D:/Master Programme/Semester Two/Artificial Intelligence within the Healthcare System/Artificial-Intelligence-within-the-Healthcare-System/Project/data/processed')
    CLASSES = ['benign', 'malignant', 'normal']
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    SEED = 42

    print("\n" + "=" * 60)
    print("  PREPROCESSING & TRAIN/VAL/TEST SPLIT")
    print("=" * 60)
    print(f"\nSplit ratios : {TRAIN_RATIO*100:.0f}% / {VAL_RATIO*100:.0f}% / {TEST_RATIO*100:.0f}%")
    print(f"Random seed  : {SEED}")

    # ─── STEP 2: Fix random seed for reproducibility ───────────
    random.seed(SEED)

    # ─── STEP 3: Create the split per class ────────────────────
    counts = {
        'train': {},
        'val': {},
        'test': {}
    }

    print("\nProcessing classes...")

    for cls in CLASSES:
        # Get all images in this class
        all_images = sorted(list((SRC_DIR / cls).glob('*.png')))
        n_total = len(all_images)

        # Shuffle randomly
        random.shuffle(all_images)

        # Calculate split sizes
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)
        n_test = n_total - n_train - n_val

        # Slice into 3 groups
        train_imgs = all_images[:n_train]
        val_imgs = all_images[n_train:n_train + n_val]
        test_imgs = all_images[n_train + n_val:]

        # Copy images to destination folders
        for split_name, img_list in [('train', train_imgs),
                                      ('val', val_imgs),
                                      ('test', test_imgs)]:
            dest_folder = DEST_DIR / split_name / cls
            dest_folder.mkdir(parents=True, exist_ok=True)

            for img_path in img_list:
                shutil.copy(img_path, dest_folder / img_path.name)

            counts[split_name][cls] = len(img_list)

        print(f"  {cls:<12} → train={n_train}  val={n_val}  test={n_test}  total={n_total}")

    # ─── STEP 4: Print summary table ───────────────────────────
    print("\n" + "=" * 70)
    print(f"  {'Split':<8}  {'Benign':>8}  {'Malignant':>10}  {'Normal':>8}  {'Total':>7}")
    print("=" * 70)

    grand_total = 0
    for split_name in ['train', 'val', 'test']:
        b = counts[split_name]['benign']
        m = counts[split_name]['malignant']
        n = counts[split_name]['normal']
        t = b + m + n
        grand_total += t
        print(f"  {split_name:<8}  {b:>8}  {m:>10}  {n:>8}  {t:>7}")

    print("=" * 70)
    print(f"  {'TOTAL':<8}  "
          f"{sum(counts[s]['benign'] for s in ['train','val','test']):>8}  "
          f"{sum(counts[s]['malignant'] for s in ['train','val','test']):>10}  "
          f"{sum(counts[s]['normal'] for s in ['train','val','test']):>8}  "
          f"{grand_total:>7}")
    print("=" * 70)

    # ─── STEP 5: Verify with PyTorch ImageFolder ───────────────
    print("\nVerifying with PyTorch ImageFolder...")

    basic_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder('D:/Master Programme/Semester Two/Artificial Intelligence within the Healthcare System/Artificial-Intelligence-within-the-Healthcare-System/Project/data/processed/train', transform=basic_transform)
    val_dataset = datasets.ImageFolder('D:/Master Programme/Semester Two/Artificial Intelligence within the Healthcare System/Artificial-Intelligence-within-the-Healthcare-System/Project/data/processed/val', transform=basic_transform)
    test_dataset = datasets.ImageFolder('D:/Master Programme/Semester Two/Artificial Intelligence within the Healthcare System/Artificial-Intelligence-within-the-Healthcare-System/Project/data/processed/test', transform=basic_transform)

    print(f"  Train dataset : {len(train_dataset)} images")
    print(f"  Val dataset   : {len(val_dataset)} images")
    print(f"  Test dataset  : {len(test_dataset)} images")
    print(f"  Classes       : {train_dataset.classes}")

    # Check one sample
    sample_image, sample_label = train_dataset[0]
    print(f"  Sample shape  : {sample_image.shape}  (3 channels, 224×224)")

    print("\n✓ Split complete and verified!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    run_preprocessing()
