"""
Script to check the augmentation ratio in existing PSL dataset folders.
"""

import os

DATASET_DIR = "data/UAlpha40 A Comprehensive Dataset of Urdu alphabets for Pakistan Sign Language"

# Pairs to check
FOLDER_PAIRS = [
    ("Hamza-Original", "Hamza-Augmented"),
    ("Alif-Original", "Alif-Augmented"),
    ("Bay-Original", "Bay-Augmented"),
    ("Ain-Original", "Ain-Augmented"),
]

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')

def count_images(folder_path):
    """Count number of images in a folder."""
    if not os.path.exists(folder_path):
        return 0
    return sum(1 for f in os.listdir(folder_path) if f.lower().endswith(IMAGE_EXTENSIONS))

print("=" * 70)
print("Checking Augmentation Ratios")
print("=" * 70 + "\n")

for original, augmented in FOLDER_PAIRS:
    orig_path = os.path.join(DATASET_DIR, original)
    aug_path = os.path.join(DATASET_DIR, augmented)
    
    orig_count = count_images(orig_path)
    aug_count = count_images(aug_path)
    
    if orig_count > 0:
        ratio = aug_count / orig_count
        print(f"{original}:")
        print(f"  Original images: {orig_count}")
        print(f"  Augmented images: {aug_count}")
        print(f"  Ratio: {ratio:.2f} augmentations per original\n")
    else:
        print(f"{original}: Folder not found or empty\n")

print("=" * 70)

