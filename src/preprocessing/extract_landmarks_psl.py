"""
Extract hand landmarks from PSL (Pakistan Sign Language) dataset images.
Processes all folders in the UAlpha40 dataset (Original + Augmented).
"""

import os
import cv2
import csv
import mediapipe as mp
from time import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------------------------------------------------------
# ⚙️ Configuration
# ---------------------------------------------------------------------
DATA_DIR = "data/UAlpha40 A Comprehensive Dataset of Urdu alphabets for Pakistan Sign Language"
OUTPUT_CSV = "data/psl_landmarks.csv"

mp_hands = mp.solutions.hands


# ---------------------------------------------------------------------
# 🖐️ Process one folder (runs in parallel)
# ---------------------------------------------------------------------
def process_folder(label_folder):
    folder_name = os.path.basename(label_folder)
    
    # Clean up label: remove -Original, -Augmented, -Generated suffixes
    # to combine them into single alphabet classes
    label = folder_name.replace("-Original", "").replace("-Augmented", "").replace("-Generated", "")
    # Also normalize some folder names (e.g., "1-Hay" -> "Hay")
    if label.startswith("1-Hay"):
        label = "Hay"
    elif label.startswith("2-Hay"):
        label = "Hay"
    
    images = [
        os.path.join(label_folder, f)
        for f in os.listdir(label_folder)
        if f.lower().endswith((".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"))
    ]

    total = len(images)
    rows, detected = [], 0
    start = time()

    # Create Mediapipe Hands instance once per process
    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3
    ) as hands:
        for idx, img_path in enumerate(images, start=1):
            image = cv2.imread(img_path)
            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0]
                row = [label]
                for p in lm.landmark:
                    row.extend([p.x, p.y])
                rows.append(row)
                detected += 1

            # Heartbeat progress log
            if idx % 500 == 0:
                print(f"[{folder_name}] {idx}/{total} ({detected} usable, {time()-start:.1f}s)", flush=True)

    print(f"✅ Finished '{folder_name}' → label '{label}': {detected}/{total} usable images\n", flush=True)
    return rows


# ---------------------------------------------------------------------
# 🚀 Main script (multi-core orchestration)
# ---------------------------------------------------------------------
def main():
    print("=" * 70)
    print("🖐️  PSL Landmark Extraction")
    print("=" * 70)
    print(f"\nDataset: {DATA_DIR}")
    print(f"Output: {OUTPUT_CSV}\n")
    print("=" * 70 + "\n")

    label_folders = [
        os.path.join(DATA_DIR, d)
        for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and not d.startswith(".")
    ]

    print(f"Found {len(label_folders)} folders to process\n")

    header = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ["x", "y"]]
    all_rows = []

    start_all = time()
    # Use maximum CPU cores for parallel processing
    max_workers = os.cpu_count()
    print(f"Using {max_workers} parallel workers (all CPU cores)\n")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_folder, folder): folder for folder in label_folders}
        for future in as_completed(futures):
            folder_name = os.path.basename(futures[future])
            try:
                rows = future.result()
                all_rows.extend(rows)
            except Exception as e:
                print(f"❌ Error in folder '{folder_name}': {e}", flush=True)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_rows)

    print("\n" + "=" * 70)
    print("📊 EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"\nTotal landmarks extracted: {len(all_rows):,}")
    print(f"Output file: {OUTPUT_CSV}")
    print(f"Total time: {time() - start_all:.1f}s")
    print("\n✅ PSL landmark extraction complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

