"""
Real-time PSL (Pakistan Sign Language) inference script.
Predicts Urdu alphabet signs from images using trained landmark model.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib
from pathlib import Path

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
MODEL_PATH = "models/psl/psl_landmark_classifier.h5"
LABEL_ENCODER_PATH = "models/psl/label_encoder.pkl"
SCALER_PATH = "models/psl/scaler.pkl"

print("=" * 70)
print("PSL (Pakistan Sign Language) Inference")
print("=" * 70)

# ---------------------------------------------------------------------
# Load model and encoders
# ---------------------------------------------------------------------
print("\nLoading model and encoders...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"[OK] Model loaded: {MODEL_PATH}")
    print(f"[OK] Classes: {len(label_encoder.classes_)} Urdu alphabets")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    exit(1)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ---------------------------------------------------------------------
# Extract landmarks from image
# ---------------------------------------------------------------------
def extract_landmarks(image_path):
    """
    Extract hand landmarks from an image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Tuple of (landmarks_array, original_image) or (None, None)
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return None, None
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:
        results = hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            print(f"[WARN] No hand detected in image")
            return None, image
        
        # Extract landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y])
        
        return np.array(landmarks).reshape(1, -1), image


# ---------------------------------------------------------------------
# Make prediction
# ---------------------------------------------------------------------
def predict_sign(image_path, show_top_n=5):
    """
    Predict PSL sign from image.
    
    Args:
        image_path: Path to image file
        show_top_n: Number of top predictions to show
        
    Returns:
        Predicted class and confidence
    """
    print(f"\n{'='*70}")
    print(f"Processing: {image_path}")
    print(f"{'='*70}")
    
    # Extract landmarks
    landmarks, original_image = extract_landmarks(image_path)
    
    if landmarks is None:
        return None, 0.0
    
    # Normalize landmarks
    landmarks_scaled = scaler.transform(landmarks)
    
    # Make prediction
    predictions = model.predict(landmarks_scaled, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    predicted_class = label_encoder.classes_[predicted_class_idx]
    
    # Display results
    print(f"\n[PREDICTION]")
    print(f"  Sign: {predicted_class}")
    print(f"  Confidence: {confidence*100:.2f}%")
    
    # Show top N predictions
    print(f"\n[TOP {show_top_n} PREDICTIONS]")
    top_indices = np.argsort(predictions[0])[::-1][:show_top_n]
    for i, idx in enumerate(top_indices, 1):
        class_name = label_encoder.classes_[idx]
        conf = predictions[0][idx]
        bar = "█" * int(conf * 30)
        print(f"  {i}. {class_name:15s} {conf*100:5.2f}% {bar}")
    
    return predicted_class, confidence


# ---------------------------------------------------------------------
# Process single image
# ---------------------------------------------------------------------
def process_single_image(image_path):
    """Process a single image."""
    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}")
        return
    
    predicted_class, confidence = predict_sign(image_path)
    
    if predicted_class:
        print(f"\n{'='*70}")
        print(f"Result: '{predicted_class}' ({confidence*100:.2f}% confidence)")
        print(f"{'='*70}")


# ---------------------------------------------------------------------
# Process folder of images
# ---------------------------------------------------------------------
def process_folder(folder_path):
    """Process all images in a folder."""
    if not os.path.exists(folder_path):
        print(f"[ERROR] Folder not found: {folder_path}")
        return
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = [
        f for f in os.listdir(folder_path)
        if Path(f).suffix in image_extensions
    ]
    
    if not image_files:
        print(f"[WARN] No images found in {folder_path}")
        return
    
    print(f"\nFound {len(image_files)} images in folder")
    
    results = []
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        predicted_class, confidence = predict_sign(img_path)
        if predicted_class:
            results.append((img_file, predicted_class, confidence))
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY - Processed {len(results)} images")
    print(f"{'='*70}")
    for img_file, pred, conf in results:
        print(f"  {img_file:40s} -> {pred:15s} ({conf*100:.1f}%)")


# ---------------------------------------------------------------------
# Main interactive menu
# ---------------------------------------------------------------------
def main():
    print("\nOptions:")
    print("  1. Process single image")
    print("  2. Process folder of images")
    print("  3. Exit")
    
    while True:
        print(f"\n{'='*70}")
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            image_path = input("Enter image path: ").strip()
            process_single_image(image_path)
            
        elif choice == "2":
            folder_path = input("Enter folder path: ").strip()
            process_folder(folder_path)
            
        elif choice == "3":
            print("\nExiting...")
            break
        else:
            print("[ERROR] Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()

