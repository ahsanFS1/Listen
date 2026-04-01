"""
Script to delete the incorrectly created augmented folders.
Removes 2-Hay-Augmented, Alifmad-Augmented, Aray-Augmented, Jeem-Augmented.
"""

import os
import shutil

DATASET_DIR = "data/UAlpha40 A Comprehensive Dataset of Urdu alphabets for Pakistan Sign Language"

FOLDERS_TO_DELETE = [
    "2-Hay-Augmented",
    "Alifmad-Augmented",
    "Aray-Augmented",
    "Jeem-Augmented"
]

def main():
    print("=" * 70)
    print("Delete Incorrectly Created Augmented Folders")
    print("=" * 70 + "\n")
    
    for folder_name in FOLDERS_TO_DELETE:
        folder_path = os.path.join(DATASET_DIR, folder_name)
        
        if os.path.exists(folder_path):
            try:
                # Count files before deletion
                num_files = len([f for f in os.listdir(folder_path) 
                                if os.path.isfile(os.path.join(folder_path, f))])
                
                # Delete the folder
                shutil.rmtree(folder_path)
                print(f"Deleted: {folder_name} ({num_files} files)")
            except Exception as e:
                print(f"Error deleting {folder_name}: {e}")
        else:
            print(f"Not found: {folder_name}")
    
    print("\n" + "=" * 70)
    print("Cleanup complete!")
    print("You can now run the updated augmentation script.")
    print("=" * 70)

if __name__ == "__main__":
    main()

