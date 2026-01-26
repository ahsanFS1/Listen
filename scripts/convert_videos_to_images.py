"""
Script to convert videos in PSL dataset to images by extracting the last frame.
This ensures all folders contain images in the same format for landmark extraction.
"""

import os
import cv2
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------
# ⚙️ Configuration
# ---------------------------------------------------------------------
DATASET_DIR = "data/UAlpha40 A Comprehensive Dataset of Urdu alphabets for Pakistan Sign Language"
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV')
IMAGE_FORMAT = '.jpg'  # Output format for extracted frames


# ---------------------------------------------------------------------
# 🎬 Extract last frame from video and save as image
# ---------------------------------------------------------------------
def extract_frame_at_percent(video_path, output_path, percent=70):
    """
    Extract a frame at a specific percentage of the video duration.
    
    Args:
        video_path: Path to the video file
        output_path: Path to save the extracted image
        percent: Percentage of video duration to extract frame from (default: 70%)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return False
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            print(f"❌ Video has no frames: {video_path}")
            cap.release()
            return False
        
        # Calculate frame position at the specified percentage
        target_frame = int(total_frames * (percent / 100.0))
        
        # Ensure we don't go beyond the video
        target_frame = min(target_frame, total_frames - 1)
        target_frame = max(target_frame, 0)
        
        # Set position to target frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        
        # Read the frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            print(f"❌ Could not read frame at {percent}%: {video_path}")
            return False
        
        # Save the frame as an image
        cv2.imwrite(output_path, frame)
        return True
        
    except Exception as e:
        print(f"❌ Error processing {video_path}: {e}")
        return False


# ---------------------------------------------------------------------
# 📁 Process a single folder
# ---------------------------------------------------------------------
def process_folder(folder_path):
    """
    Process all videos in a folder and extract last frames.
    
    Args:
        folder_path: Path to the folder containing videos
        
    Returns:
        Tuple of (folder_name, total_videos, successful_conversions)
    """
    folder_name = os.path.basename(folder_path)
    video_files = []
    
    # Find all video files in the folder
    for file in os.listdir(folder_path):
        if file.lower().endswith(VIDEO_EXTENSIONS):
            video_files.append(file)
    
    if not video_files:
        return folder_name, 0, 0
    
    successful = 0
    
    for video_file in tqdm(video_files, desc=f"Processing {folder_name}", leave=False):
        video_path = os.path.join(folder_path, video_file)
        
        # Create output image filename (replace video extension with .jpg)
        image_filename = os.path.splitext(video_file)[0] + IMAGE_FORMAT
        image_path = os.path.join(folder_path, image_filename)
        
        # Skip if image already exists
        if os.path.exists(image_path):
            print(f"⏩ Skipping {video_file} - image already exists")
            successful += 1
            continue
        
        # Extract frame at 70% of video duration
        if extract_frame_at_percent(video_path, image_path, percent=70):
            successful += 1
    
    return folder_name, len(video_files), successful


# ---------------------------------------------------------------------
# 🔍 Find folders containing videos
# ---------------------------------------------------------------------
def find_video_folders(dataset_dir):
    """
    Scan dataset directory and find all folders containing videos.
    
    Args:
        dataset_dir: Path to the dataset root directory
        
    Returns:
        List of folder paths containing videos
    """
    video_folders = []
    
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return video_folders
    
    for folder_name in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
        
        # Check if folder contains any videos
        has_videos = False
        for file in os.listdir(folder_path):
            if file.lower().endswith(VIDEO_EXTENSIONS):
                has_videos = True
                break
        
        if has_videos:
            video_folders.append(folder_path)
    
    return video_folders


# ---------------------------------------------------------------------
# 🚀 Main script
# ---------------------------------------------------------------------
def main():
    print("=" * 70)
    print("🎬 PSL Video to Image Converter")
    print("=" * 70)
    print(f"\nDataset directory: {DATASET_DIR}")
    print(f"Extracting frames at 70% of video duration\n")
    
    # Find all folders containing videos
    print("🔍 Scanning for folders with videos...")
    video_folders = find_video_folders(DATASET_DIR)
    
    if not video_folders:
        print("✅ No folders with videos found. All folders already contain images only.")
        return
    
    print(f"\n📊 Found {len(video_folders)} folder(s) containing videos:")
    for folder in video_folders:
        folder_name = os.path.basename(folder)
        video_count = sum(1 for f in os.listdir(folder) if f.lower().endswith(VIDEO_EXTENSIONS))
        print(f"  • {folder_name}: {video_count} videos")
    
    print("\n" + "=" * 70)
    print("🎯 Starting conversion process...")
    print("=" * 70 + "\n")
    
    # Process each folder
    total_videos = 0
    total_successful = 0
    results = []
    
    for folder_path in video_folders:
        folder_name, num_videos, num_successful = process_folder(folder_path)
        results.append((folder_name, num_videos, num_successful))
        total_videos += num_videos
        total_successful += num_successful
        print(f"✅ {folder_name}: {num_successful}/{num_videos} videos converted successfully")
    
    # Summary
    print("\n" + "=" * 70)
    print("📈 CONVERSION SUMMARY")
    print("=" * 70)
    print(f"\nTotal folders processed: {len(video_folders)}")
    print(f"Total videos found: {total_videos}")
    print(f"Successfully converted: {total_successful}")
    print(f"Failed: {total_videos - total_successful}")
    
    if total_successful == total_videos:
        print("\n✅ All videos successfully converted to images!")
    else:
        print(f"\n⚠️  {total_videos - total_successful} videos failed to convert.")
    
    print("\n" + "=" * 70)
    print("🎉 Conversion complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

