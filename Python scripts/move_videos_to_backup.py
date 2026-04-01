"""
Script to move video files from PSL dataset folders to a backup location.
This cleans up the dataset folders so they only contain images.
"""

import os
import shutil
from pathlib import Path

# ---------------------------------------------------------------------
# ⚙️ Configuration
# ---------------------------------------------------------------------
DATASET_DIR = "data/UAlpha40 A Comprehensive Dataset of Urdu alphabets for Pakistan Sign Language"
BACKUP_DIR = "data/psl_videos_backup"
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV')

# Folders that contain videos
VIDEO_FOLDERS = [
    "2-Hay",
    "Alifmad",
    "Aray",
    "Jeem"
]


# ---------------------------------------------------------------------
# 📦 Move videos from a folder
# ---------------------------------------------------------------------
def move_videos_from_folder(folder_path, backup_subfolder):
    """
    Move all video files from a folder to the backup location.
    
    Args:
        folder_path: Path to the source folder
        backup_subfolder: Path to the backup subfolder
        
    Returns:
        Number of videos moved
    """
    if not os.path.exists(folder_path):
        print(f"⚠️  Folder not found: {folder_path}")
        return 0
    
    # Create backup subfolder if it doesn't exist
    os.makedirs(backup_subfolder, exist_ok=True)
    
    moved_count = 0
    
    # Find and move all video files
    for file in os.listdir(folder_path):
        if file.lower().endswith(VIDEO_EXTENSIONS):
            source_path = os.path.join(folder_path, file)
            dest_path = os.path.join(backup_subfolder, file)
            
            try:
                shutil.move(source_path, dest_path)
                moved_count += 1
                print(f"  📦 Moved: {file}")
            except Exception as e:
                print(f"  ❌ Error moving {file}: {e}")
    
    return moved_count


# ---------------------------------------------------------------------
# 🚀 Main script
# ---------------------------------------------------------------------
def main():
    print("=" * 70)
    print("📦 Move PSL Videos to Backup")
    print("=" * 70)
    print(f"\nDataset directory: {DATASET_DIR}")
    print(f"Backup directory: {BACKUP_DIR}\n")
    print("=" * 70 + "\n")
    
    # Create main backup directory
    os.makedirs(BACKUP_DIR, exist_ok=True)
    
    total_moved = 0
    
    for folder_name in VIDEO_FOLDERS:
        folder_path = os.path.join(DATASET_DIR, folder_name)
        backup_subfolder = os.path.join(BACKUP_DIR, folder_name)
        
        print(f"📁 Processing folder: {folder_name}")
        
        moved = move_videos_from_folder(folder_path, backup_subfolder)
        total_moved += moved
        
        if moved > 0:
            print(f"✅ Moved {moved} video(s) from {folder_name}\n")
        else:
            print(f"ℹ️  No videos found in {folder_name}\n")
    
    print("=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"\nTotal videos moved: {total_moved}")
    print(f"Backup location: {BACKUP_DIR}")
    
    if total_moved > 0:
        print("\n✅ All videos successfully moved to backup!")
        print("   Dataset folders now contain only images.")
    else:
        print("\nℹ️  No videos found to move.")
    
    print("=" * 70)


if __name__ == "__main__":
    main()

