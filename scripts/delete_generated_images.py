"""
Script to delete generated .jpg images from video folders in PSL dataset.
This removes the incorrectly extracted black frames so we can re-extract at 70%.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------
# ⚙️ Configuration
# ---------------------------------------------------------------------
DATASET_DIR = "data/UAlpha40 A Comprehensive Dataset of Urdu alphabets for Pakistan Sign Language"

# Folders that contain videos (and where we generated .jpg files)
VIDEO_FOLDERS = [
    "2-Hay",
    "Alifmad",
    "Aray",
    "Jeem"
]


# ---------------------------------------------------------------------
# 🗑️ Delete generated images
# ---------------------------------------------------------------------
def delete_images_in_folder(folder_path):
    """
    Delete all .jpg files in a folder that has corresponding .mp4 files.
    Only deletes if a video with the same name exists.
    
    Args:
        folder_path: Path to the folder
        
    Returns:
        Number of images deleted
    """
    if not os.path.exists(folder_path):
        print(f"⚠️  Folder not found: {folder_path}")
        return 0
    
    deleted_count = 0
    
    # Get list of all files
    files = os.listdir(folder_path)
    
    # Find .mp4 files
    video_basenames = set()
    for file in files:
        if file.lower().endswith('.mp4'):
            basename = os.path.splitext(file)[0]
            video_basenames.add(basename)
    
    # Delete .jpg files that have corresponding videos
    for file in files:
        if file.lower().endswith('.jpg'):
            basename = os.path.splitext(file)[0]
            if basename in video_basenames:
                image_path = os.path.join(folder_path, file)
                try:
                    os.remove(image_path)
                    deleted_count += 1
                    print(f"  🗑️  Deleted: {file}")
                except Exception as e:
                    print(f"  ❌ Error deleting {file}: {e}")
    
    return deleted_count


# ---------------------------------------------------------------------
# 🚀 Main script
# ---------------------------------------------------------------------
def main():
    print("=" * 70)
    print("🗑️  Delete Generated Images from Video Folders")
    print("=" * 70)
    print(f"\nDataset directory: {DATASET_DIR}\n")
    
    total_deleted = 0
    
    for folder_name in VIDEO_FOLDERS:
        folder_path = os.path.join(DATASET_DIR, folder_name)
        print(f"\n📁 Processing folder: {folder_name}")
        
        deleted = delete_images_in_folder(folder_path)
        total_deleted += deleted
        
        if deleted > 0:
            print(f"✅ Deleted {deleted} image(s) from {folder_name}")
        else:
            print(f"ℹ️  No images to delete in {folder_name}")
    
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"\nTotal images deleted: {total_deleted}")
    print("\n✅ Cleanup complete! You can now run the converter script again.")
    print("=" * 70)


if __name__ == "__main__":
    main()

