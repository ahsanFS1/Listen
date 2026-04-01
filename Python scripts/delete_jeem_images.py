"""
Script to delete generated .jpg images from Jeem folder only.
"""

import os

# ---------------------------------------------------------------------
# ⚙️ Configuration
# ---------------------------------------------------------------------
JEEM_FOLDER = "data/UAlpha40 A Comprehensive Dataset of Urdu alphabets for Pakistan Sign Language/Jeem"


# ---------------------------------------------------------------------
# 🗑️ Delete generated images
# ---------------------------------------------------------------------
def delete_images_in_jeem():
    """
    Delete all .jpg files in Jeem folder that have corresponding .mp4 files.
    
    Returns:
        Number of images deleted
    """
    if not os.path.exists(JEEM_FOLDER):
        print(f"⚠️  Folder not found: {JEEM_FOLDER}")
        return 0
    
    deleted_count = 0
    
    # Get list of all files
    files = os.listdir(JEEM_FOLDER)
    
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
                image_path = os.path.join(JEEM_FOLDER, file)
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
    print("🗑️  Delete Generated Images from Jeem Folder")
    print("=" * 70)
    print(f"\nFolder: {JEEM_FOLDER}\n")
    
    deleted = delete_images_in_jeem()
    
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"\nTotal images deleted: {deleted}")
    
    if deleted > 0:
        print("\n✅ Cleanup complete! You can now run the Jeem converter script.")
    else:
        print("\nℹ️  No images found to delete.")
    
    print("=" * 70)


if __name__ == "__main__":
    main()

