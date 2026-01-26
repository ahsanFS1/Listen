"""
Script to analyze the augmentation pattern in existing folders.
Checks which images are augmented and how many augmentations each has.
"""

import os
from collections import defaultdict

DATASET_DIR = "data/UAlpha40 A Comprehensive Dataset of Urdu alphabets for Pakistan Sign Language"

# Check a few folder pairs
FOLDER_PAIRS = [
    ("Zaey-Original", "Zaey-Augmented"),
    ("Hamza-Original", "Hamza-Augmented"),
]

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')

def analyze_augmentation(original_folder, augmented_folder):
    """Analyze augmentation pattern."""
    orig_path = os.path.join(DATASET_DIR, original_folder)
    aug_path = os.path.join(DATASET_DIR, augmented_folder)
    
    if not os.path.exists(orig_path) or not os.path.exists(aug_path):
        print(f"Folders not found\n")
        return
    
    # Count original images
    orig_images = [f for f in os.listdir(orig_path) if f.lower().endswith(IMAGE_EXTENSIONS)]
    
    # Analyze augmented images
    aug_images = [f for f in os.listdir(aug_path) if f.lower().endswith(IMAGE_EXTENSIONS)]
    
    # Group augmented images by their base name
    aug_counts = defaultdict(int)
    for aug_img in aug_images:
        # Try to extract base name (before -Generated or -aug)
        if "-Generated-" in aug_img or "-generated-" in aug_img.lower():
            base = aug_img.split("-Generated-")[0] if "-Generated-" in aug_img else aug_img.split("-generated-")[0]
        elif "-aug" in aug_img:
            base = aug_img.split("-aug")[0]
        else:
            continue
        aug_counts[base] += 1
    
    print(f"{original_folder} -> {augmented_folder}:")
    print(f"  Total original images: {len(orig_images)}")
    print(f"  Total augmented images: {len(aug_images)}")
    print(f"  Number of images that were augmented: {len(aug_counts)}")
    
    if len(aug_counts) > 0:
        avg_aug = sum(aug_counts.values()) / len(aug_counts)
        print(f"  Average augmentations per augmented image: {avg_aug:.1f}")
        print(f"  Percentage of images augmented: {100 * len(aug_counts) / len(orig_images):.1f}%")
        
        # Show some examples
        print(f"\n  Sample augmentation counts:")
        for i, (base, count) in enumerate(list(aug_counts.items())[:5]):
            print(f"    {base}: {count} augmentations")
    
    print()

print("=" * 70)
print("Analyzing Augmentation Pattern")
print("=" * 70 + "\n")

for original, augmented in FOLDER_PAIRS:
    analyze_augmentation(original, augmented)

print("=" * 70)

