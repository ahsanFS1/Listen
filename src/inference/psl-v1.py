"""
📷 Listen - Real-time PSL (Pakistan Sign Language) Inference
Live Urdu alphabet recognition with word accumulation.
"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
import os
import time

# ---------------------------------------------------------------------
# ⚙️ Configuration
# ---------------------------------------------------------------------
MODEL_PATH = "models/psl/psl_landmark_classifier.tflite"
ENCODER_PATH = "data/psl_processed/label_encoder.pkl"
SCALER_PATH = "data/psl_processed/scaler.pkl"

# Load model, encoder, and scaler
print("Loading PSL model and encoders...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
if not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError(f"Encoder not found at {ENCODER_PATH}")
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
label_encoder = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)

print(f"[OK] Model loaded: {len(label_encoder.classes_)} Urdu alphabet classes")

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------------------------------------------------------------
# 🔮 Helper: Predict from landmarks
# ---------------------------------------------------------------------
def predict_sign(landmarks):
    """
    Predict PSL sign from hand landmarks.
    
    Args:
        landmarks: Flattened array of 42 values (21 x,y coordinates)
        
    Returns:
        Tuple of (predicted_label, confidence)
    """
    # Reshape and scale landmarks
    landmarks = np.array(landmarks, dtype=np.float32).reshape(1, -1)
    landmarks_scaled = scaler.transform(landmarks)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], landmarks_scaled.astype(np.float32))
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    
    label_idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    label = label_encoder.inverse_transform([label_idx])[0]
    
    return label, confidence

# ---------------------------------------------------------------------
# 🧠 Word accumulation logic
# ---------------------------------------------------------------------
def update_text(pred_label, confidence, last_label, stable_count, sentence):
    """
    Update accumulated sentence based on stable predictions.
    
    Args:
        pred_label: Current predicted label
        confidence: Prediction confidence
        last_label: Previous label
        stable_count: Number of consecutive frames with same label
        sentence: Current accumulated text
        
    Returns:
        Tuple of (new_last_label, new_stable_count, new_sentence)
    """
    THRESHOLD = 0.85  # confidence threshold for PSL
    STABLE_REQUIRED = 10  # number of stable frames before accepting letter
    
    if confidence > THRESHOLD:
        if pred_label == last_label:
            stable_count += 1
        else:
            stable_count = 0
        
        if stable_count == STABLE_REQUIRED:
            # Add the Urdu alphabet to sentence
            sentence += pred_label + " "
            stable_count = 0  # reset after adding letter
    
    return last_label if confidence <= THRESHOLD else pred_label, stable_count, sentence

# ---------------------------------------------------------------------
# 🎥 Real-time inference
# ---------------------------------------------------------------------
def run_inference():
    """Run real-time PSL inference from webcam."""
    print("\n" + "="*70)
    print("Starting PSL Real-Time Recognition")
    print("="*70)
    print("\nControls:")
    print("  'q' - Quit")
    print("  'c' - Clear sentence")
    print("  'Space' - Add space to sentence")
    print("\nShow a PSL sign to the camera...")
    print("="*70 + "\n")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Could not access webcam.")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    sentence = ""
    last_label = None
    stable_count = 0
    fps_time = time.time()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)  # Mirror the frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        h, w, _ = frame.shape
        
        # Draw semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks with style
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Extract 21 (x, y) landmarks
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])
                
                # Predict
                label, conf = predict_sign(landmarks)
                last_label, stable_count, sentence = update_text(
                    label, conf, last_label, stable_count, sentence
                )
                
                # Display current prediction
                pred_text = f"{label} ({conf*100:.1f}%)"
                color = (0, 255, 0) if conf > 0.85 else (0, 165, 255)
                cv2.putText(frame, pred_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                           1.2, color, 3, cv2.LINE_AA)
                
                # Stability indicator
                stability_bar = int((stable_count / 10) * 200)
                cv2.rectangle(frame, (20, 70), (20 + stability_bar, 85),
                            (0, 255, 0), -1)
                cv2.rectangle(frame, (20, 70), (220, 85), (255, 255, 255), 2)
        else:
            # No hand detected
            cv2.putText(frame, "No hand detected", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Display accumulated sentence
        cv2.putText(frame, "Recognized:", (20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Word wrap for long sentences
        display_sentence = sentence if len(sentence) < 50 else "..." + sentence[-47:]
        cv2.putText(frame, display_sentence, (20, 155),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)
        
        # FPS counter
        frame_count += 1
        if frame_count % 10 == 0:
            fps = 10.0 / (time.time() - fps_time)
            fps_time = time.time()
            current_fps = fps
        else:
            current_fps = 0 if frame_count < 10 else current_fps
        
        if frame_count >= 10:
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (w - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit | 'c' to clear | 'space' for space",
                   (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.imshow("Listen - PSL Real-Time Recognition", frame)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            sentence = ""
            print("Sentence cleared")
        elif key == ord(' '):
            sentence += " | "
            print("Space added")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*70)
    print("Inference stopped")
    print("="*70)
    print(f"\nFinal sentence: {sentence}")
    print("="*70 + "\n")


# ---------------------------------------------------------------------
# 🚀 Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    try:
        run_inference()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

