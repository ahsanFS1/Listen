"""
📷 Listen - Real-time ASL Inference
Uses your trained TFLite model + MediaPipe Hands for live camera prediction.
"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
import os

# ---------------------------------------------------------------------
# ⚙️ Configuration
# ---------------------------------------------------------------------
MODEL_PATH = "saved_models/asl_landmark_classifier.tflite"
ENCODER_PATH = "data/processed/label_encoder.pkl"

# Load model and encoder
print("🔹 Loading model and label encoder...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
if not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError(f"Encoder not found at {ENCODER_PATH}")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
label_encoder = joblib.load(ENCODER_PATH)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.3)

# ---------------------------------------------------------------------
# 🔮 Helper: Predict from landmarks
# ---------------------------------------------------------------------
def predict_sign(landmarks):
    landmarks = np.array(landmarks, dtype=np.float32).reshape(1, -1)
    interpreter.set_tensor(input_details[0]['index'], landmarks)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    label_idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    label = label_encoder.inverse_transform([label_idx])[0]
    return label, confidence

# ---------------------------------------------------------------------
# 🎥 Real-time inference
# ---------------------------------------------------------------------
def run_inference():
    print("🎥 Starting webcam... Press 'q' to quit.")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Could not access webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror view
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract 21 (x, y)
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])

                label, conf = predict_sign(landmarks)
                text = f"{label} ({conf*100:.1f}%)"
                cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Listen - ASL Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Inference stopped.")


# ---------------------------------------------------------------------
# 🚀 Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    run_inference()
