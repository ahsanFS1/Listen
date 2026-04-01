"""
Script to convert Jeem videos to images by extracting the FIRST frame.
"""

import os
import cv2
from tqdm import tqdm

# ---------------------------------------------------------------------
# ⚙️ Configuration
# ---------------------------------------------------------------------
JEEM_FOLDER = "data/UAlpha40 A Comprehensive Dataset of Urdu alphabets for Pakistan Sign Language/Jeem"
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV')
IMAGE_FORMAT = '.jpg'


# ---------------------------------------------------------------------
# 🎬 Extract first frame from video
# ---------------------------------------------------------------------
def extract_first_frame(video_path, output_path):
    """
    Extract the first frame from a video and save it as an image.
    
    Args:
        video_path: Path to the video file
        output_path: Path to save the extracted image
        
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
        
        # Set position to first frame (frame 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Read the first frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            print(f"❌ Could not read first frame: {video_path}")
            return False
        
        # Save the frame as an image
        cv2.imwrite(output_path, frame)
        return True
        
    except Exception as e:
        print(f"❌ Error processing {video_path}: {e}")
        return False


# ---------------------------------------------------------------------
# 📁 Process Jeem folder
# ---------------------------------------------------------------------
def process_jeem_folder():
    """
    Process all videos in Jeem folder and extract first frames.
    
    Returns:
        Tuple of (total_videos, successful_conversions)
    """
    if not os.path.exists(JEEM_FOLDER):
        print(f"❌ Folder not found: {JEEM_FOLDER}")
        return 0, 0
    
    video_files = []
    
    # Find all video files in the folder
    for file in os.listdir(JEEM_FOLDER):
        if file.lower().endswith(VIDEO_EXTENSIONS):
            video_files.append(file)
    
    if not video_files:
        print("ℹ️  No video files found in Jeem folder.")
        return 0, 0
    
    print(f"Found {len(video_files)} videos in Jeem folder\n")
    
    successful = 0
    
    for video_file in tqdm(video_files, desc="Processing Jeem videos", unit="video"):
        video_path = os.path.join(JEEM_FOLDER, video_file)
        
        # Create output image filename
        image_filename = os.path.splitext(video_file)[0] + IMAGE_FORMAT
        image_path = os.path.join(JEEM_FOLDER, image_filename)
        
        # Skip if image already exists
        if os.path.exists(image_path):
            print(f"⏩ Skipping {video_file} - image already exists")
            successful += 1
            continue
        
        # Extract first frame
        if extract_first_frame(video_path, image_path):
            successful += 1
    
    return len(video_files), successful


# ---------------------------------------------------------------------
# 🚀 Main script
# ---------------------------------------------------------------------
def main():
    print("=" * 70)
    print("🎬 Jeem First Frame Extractor")
    print("=" * 70)
    print(f"\nFolder: {JEEM_FOLDER}")
    print("Extracting FIRST frame from each video\n")
    print("=" * 70 + "\n")
    
    # Process Jeem folder
    total_videos, successful = process_jeem_folder()
    
    # Summary
    print("\n" + "=" * 70)
    print("📈 CONVERSION SUMMARY")
    print("=" * 70)
    print(f"\nTotal videos found: {total_videos}")
    print(f"Successfully converted: {successful}")
    print(f"Failed: {total_videos - successful}")
    
    if total_videos > 0 and successful == total_videos:
        print("\n✅ All Jeem videos successfully converted to images!")
    elif successful > 0:
        print(f"\n⚠️  {total_videos - successful} videos failed to convert.")
    
    print("\n" + "=" * 70)
    print("🎉 Conversion complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

