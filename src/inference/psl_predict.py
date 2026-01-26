"""
Simple command-line PSL inference for quick testing.
Usage: python src/inference/psl_predict.py <image_path>
"""

import sys
import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib

# Configuration
MODEL_PATH = "models/psl/psl_landmark_classifier.h5"
LABEL_ENCODER_PATH = "data/psl_processed/label_encoder.pkl"
SCALER_PATH = "data/psl_processed/scaler.pkl"

def predict_psl_sign(image_path):
    """Quick PSL prediction from image."""
    
    # Load model and encoders
    model = tf.keras.models.load_model(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image '{image_path}'")
        return None
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Extract landmarks
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:
        results = hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            print("Warning: No hand detected in image")
            return None
        
        # Extract landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y])
        
        landmarks = np.array(landmarks).reshape(1, -1)
    
    # Normalize and predict
    landmarks_scaled = scaler.transform(landmarks)
    predictions = model.predict(landmarks_scaled, verbose=0)
    
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    predicted_class = label_encoder.classes_[predicted_class_idx]
    
    # Display result
    print(f"\n{'='*50}")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"{'='*50}")
    
    # Show top 3
    print("\nTop 3 predictions:")
    top_indices = np.argsort(predictions[0])[::-1][:3]
    for i, idx in enumerate(top_indices, 1):
        class_name = label_encoder.classes_[idx]
        conf = predictions[0][idx]
        print(f"  {i}. {class_name:15s} - {conf*100:5.2f}%")
    
    return predicted_class, confidence


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/inference/psl_predict.py <image_path>")
        print("\nExample:")
        print('  python src/inference/psl_predict.py "data/test_image.jpg"')
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        sys.exit(1)
    
    predict_psl_sign(image_path)

