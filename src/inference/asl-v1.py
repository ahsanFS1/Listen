"""
📷 Listen - Real-time ASL Inference
Now with live word accumulation for smoother text output.
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
MODEL_PATH = "src/saved_models/asl_landmark_classifier.tflite"
ENCODER_PATH = "src/data/processed/label_encoder.pkl"

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
# 🧠 Word accumulation logic
# ---------------------------------------------------------------------
def update_text(pred_label, confidence, last_label, stable_count, sentence):
    THRESHOLD = 0.9  # confidence threshold
    STABLE_REQUIRED = 8  # number of stable frames before accepting letter

    if confidence > THRESHOLD:
        if pred_label == last_label:
            stable_count += 1
        else:
            stable_count = 0

        if stable_count == STABLE_REQUIRED:
            if pred_label == "space":
                sentence += " "
            elif pred_label == "del" and len(sentence) > 0:
                sentence = sentence[:-1]
            else:
                sentence += pred_label
            stable_count = 0  # reset after adding letter

    return last_label if confidence <= THRESHOLD else pred_label, stable_count, sentence

# ---------------------------------------------------------------------
# 🎥 Real-time inference
# ---------------------------------------------------------------------
def run_inference():
    print("🎥 Starting webcam... Press 'q' to quit.")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Could not access webcam.")
        return

    sentence = ""
    last_label = None
    stable_count = 0
    fps_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract 21 (x, y)
                landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])

                # Normalize
                landmarks -= landmarks[0]
                max_val = np.max(np.abs(landmarks))
                if max_val > 0:
                    landmarks /= max_val
                landmarks = landmarks.flatten()

                label, conf = predict_sign(landmarks)
                last_label, stable_count, sentence = update_text(label, conf, last_label, stable_count, sentence)

                text = f"{label} ({conf*100:.1f}%)"
                cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw accumulated text
        cv2.putText(frame, f"📝 {sentence}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2, cv2.LINE_AA)

        # FPS display (optional)
        fps = 1.0 / (time.time() - fps_time)
        fps_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (100, 255, 100), 2)

        cv2.imshow("Listen - ASL Inference (Word Mode)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Inference stopped.")
    print("Final sentence:", sentence)


# ---------------------------------------------------------------------
# 🚀 Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    run_inference()
