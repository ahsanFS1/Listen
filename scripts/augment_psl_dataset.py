"""
Script to generate augmented images for PSL dataset.
Creates 24 augmented versions of each original image using various transformations.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

# ---------------------------------------------------------------------
# ⚙️ Configuration
# ---------------------------------------------------------------------
DATASET_DIR = "data/UAlpha40 A Comprehensive Dataset of Urdu alphabets for Pakistan Sign Language"
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
AUGMENTATIONS_PER_IMAGE = 24
PERCENTAGE_TO_AUGMENT = 0.30  # Augment 30% of images (like existing folders)

# Folders to process (only the newly extracted video folders)
FOLDERS_TO_AUGMENT = [
    "2-Hay",
    "Alifmad",
    "Aray",
    "Jeem"
]


# ---------------------------------------------------------------------
# 🎨 Augmentation Functions
# ---------------------------------------------------------------------
def rotate_image(image, angle):
    """Rotate image by a given angle."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def adjust_brightness(image, factor):
    """Adjust brightness by a factor."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def adjust_contrast(image, factor):
    """Adjust contrast by a factor."""
    mean = np.mean(image)
    return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)


def flip_horizontal(image):
    """Flip image horizontally."""
    return cv2.flip(image, 1)


def scale_image(image, scale):
    """Scale image by a factor."""
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    scaled = cv2.resize(image, (new_w, new_h))
    
    if scale > 1.0:
        # Crop center
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        return scaled[start_h:start_h + h, start_w:start_w + w]
    else:
        # Pad to original size
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        return cv2.copyMakeBorder(scaled, pad_h, h - new_h - pad_h, 
                                  pad_w, w - new_w - pad_w, 
                                  cv2.BORDER_REFLECT)


def add_gaussian_noise(image, sigma):
    """Add Gaussian noise to image."""
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def shift_image(image, shift_x, shift_y):
    """Shift image by x and y pixels."""
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), 
                         borderMode=cv2.BORDER_REFLECT)


# ---------------------------------------------------------------------
# 🎲 Generate augmentation combinations
# ---------------------------------------------------------------------
def generate_augmentation_params():
    """Generate 24 different augmentation parameter sets."""
    augmentations = []
    
    # Rotations (4)
    for angle in [-15, -7, 7, 15]:
        augmentations.append({'rotate': angle})
    
    # Brightness adjustments (4)
    for brightness in [0.7, 0.85, 1.15, 1.3]:
        augmentations.append({'brightness': brightness})
    
    # Flips and rotations combined (4)
    for angle in [-10, 0, 10, 20]:
        augmentations.append({'flip': True, 'rotate': angle})
    
    # Scale variations (4)
    for scale in [0.85, 0.92, 1.08, 1.15]:
        augmentations.append({'scale': scale})
    
    # Contrast adjustments (3)
    for contrast in [0.8, 1.2, 1.4]:
        augmentations.append({'contrast': contrast})
    
    # Shifts (3)
    for shift in [(-15, -10), (15, 10), (-10, 15)]:
        augmentations.append({'shift_x': shift[0], 'shift_y': shift[1]})
    
    # Combined transformations (2)
    augmentations.append({'rotate': 10, 'brightness': 1.2, 'scale': 1.05})
    augmentations.append({'rotate': -10, 'brightness': 0.85, 'scale': 0.95})
    
    return augmentations[:AUGMENTATIONS_PER_IMAGE]


# ---------------------------------------------------------------------
# 🖼️ Apply augmentation
# ---------------------------------------------------------------------
def apply_augmentation(image, params):
    """Apply augmentation based on parameters."""
    result = image.copy()
    
    if 'flip' in params and params['flip']:
        result = flip_horizontal(result)
    
    if 'rotate' in params:
        result = rotate_image(result, params['rotate'])
    
    if 'scale' in params:
        result = scale_image(result, params['scale'])
    
    if 'shift_x' in params and 'shift_y' in params:
        result = shift_image(result, params['shift_x'], params['shift_y'])
    
    if 'brightness' in params:
        result = adjust_brightness(result, params['brightness'])
    
    if 'contrast' in params:
        result = adjust_contrast(result, params['contrast'])
    
    if 'noise' in params:
        result = add_gaussian_noise(result, params['noise'])
    
    return result


# ---------------------------------------------------------------------
# 📁 Process one folder
# ---------------------------------------------------------------------
def process_folder(folder_name):
    """
    Process all images in a folder and create augmented versions.
    
    Args:
        folder_name: Name of the folder to process
        
    Returns:
        Tuple of (folder_name, total_images, augmentations_created)
    """
    original_folder = os.path.join(DATASET_DIR, folder_name)
    
    # Create augmented folder name (always add -Augmented)
    augmented_folder_name = f"{folder_name}-Augmented"
    augmented_folder = os.path.join(DATASET_DIR, augmented_folder_name)
    
    # Create augmented folder
    os.makedirs(augmented_folder, exist_ok=True)
    
    # Get all images
    image_files = [f for f in os.listdir(original_folder) 
                   if f.lower().endswith(IMAGE_EXTENSIONS)]
    
    if not image_files:
        return folder_name, 0, 0
    
    # Randomly select a subset of images to augment (30%)
    random.seed(42)  # For reproducibility
    num_to_augment = max(1, int(len(image_files) * PERCENTAGE_TO_AUGMENT))
    selected_images = random.sample(image_files, num_to_augment)
    
    # Generate augmentation parameters
    aug_params = generate_augmentation_params()
    
    total_augmentations = 0
    
    print(f"  Augmenting {len(selected_images)}/{len(image_files)} images from {folder_name}")
    
    for img_file in tqdm(selected_images, desc=f"Augmenting {folder_name}", leave=False):
        img_path = os.path.join(original_folder, img_file)
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        # Generate augmented images
        base_name = os.path.splitext(img_file)[0]
        
        for idx, params in enumerate(aug_params, start=1):
            augmented_image = apply_augmentation(image, params)
            
            # Save augmented image with naming pattern matching existing folders
            aug_filename = f"{base_name}-Generated-{idx}.jpg"
            aug_path = os.path.join(augmented_folder, aug_filename)
            
            cv2.imwrite(aug_path, augmented_image)
            total_augmentations += 1
    
    return folder_name, len(image_files), total_augmentations


# ---------------------------------------------------------------------
# 🚀 Main script
# ---------------------------------------------------------------------
def main():
    print("=" * 70)
    print("🎨 PSL Dataset Augmentation Generator")
    print("=" * 70)
    print(f"\nDataset directory: {DATASET_DIR}")
    print(f"Augmentations per selected image: {AUGMENTATIONS_PER_IMAGE}")
    print(f"Percentage of images to augment: {int(PERCENTAGE_TO_AUGMENT * 100)}%")
    print("\nProcessing folders: 2-Hay, Alifmad, Aray, Jeem")
    print("(Folders with newly extracted images from videos)\n")
    print("=" * 70 + "\n")
    
    # Check which folders exist
    existing_folders = []
    for folder_name in FOLDERS_TO_AUGMENT:
        folder_path = os.path.join(DATASET_DIR, folder_name)
        if os.path.exists(folder_path):
            existing_folders.append(folder_name)
    
    print(f"Found {len(existing_folders)} folders to process\n")
    
    total_images = 0
    total_augmentations = 0
    
    # Process each folder
    for folder_name in existing_folders:
        folder, num_images, num_aug = process_folder(folder_name)
        total_images += num_images
        total_augmentations += num_aug
        print(f"✅ {folder}: {num_images} images → {num_aug} augmentations created")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 AUGMENTATION SUMMARY")
    print("=" * 70)
    print(f"\nFolders processed: {len(existing_folders)}")
    print(f"Original images: {total_images}")
    print(f"Augmented images created: {total_augmentations}")
    print(f"Total images in dataset: {total_images + total_augmentations}")
    
    print("\n✅ Augmentation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

