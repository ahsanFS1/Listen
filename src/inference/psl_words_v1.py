"""
Real-time word-level PSL inference.

Uses MediaPipe Hands (up to two hands, with handedness) to build the
same 126-D per-frame vector as the training data, maintains a rolling
60-frame buffer, and runs the trained sequence model every `STRIDE`
frames. A word is committed when it wins `K_CONSECUTIVE` windows in a
row with high smoothed confidence, and is separated from the previous
word by an idle (`nothing` / `test_word`) window or a cooldown.
"""

import os
import time
import threading
import tempfile
import asyncio
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib

from PIL import Image, ImageDraw, ImageFont

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_SUPPORT = True
except ImportError:
    ARABIC_SUPPORT = False
    print("[WARN] arabic-reshaper / python-bidi not installed; Urdu may not render correctly.")

try:
    import edge_tts
    import pygame
    pygame.mixer.init()
    TTS_SUPPORT = True
    VOICE = "ur-PK-UzmaNeural"
except Exception as e:
    TTS_SUPPORT = False
    print(f"[WARN] TTS unavailable: {e}")

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "psl_words", "psl_word_classifier.tflite")
ENCODER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "psl_words", "label_encoder.pkl")

# ---------------------------------------------------------------------
# Runtime parameters (§4 of plan.md)
# ---------------------------------------------------------------------
T = 60                # window length in frames (must match training)
F = 126               # feature dim
STRIDE = 5            # run model every STRIDE frames
K_CONSECUTIVE = 3     # same class must win K windows in a row
EMA_ALPHA = 0.6       # softmax smoothing
COMMIT_CONF = 0.80    # smoothed confidence required to commit
COMMIT_COOLDOWN_S = 1.0  # seconds between committed words
IDLE_CLASSES = {"nothing", "test_word"}
MOTION_VAR_MIN = 1e-4    # wrist-variance gate; below this we treat as idle

# ---------------------------------------------------------------------
# Urdu mapping for the word classes (extend as needed)
# ---------------------------------------------------------------------
PSL_WORD_TO_URDU = {
    "absolutely": "بالکل",
    "aircrash": "ہوائی حادثہ",
    "airplane": "ہوائی جہاز",
    "all": "سب",
    "also": "بھی",
    "arrival": "آمد",
    "assalam-o-alaikum": "السلام علیکم",
    "atm": "اے ٹی ایم",
    "bald": "گنجا",
    "beach": "ساحل",
    "beak": "چونچ",
    "bear": "ریچھ",
    "beard": "داڑھی",
    "bed": "بستر",
    "bench": "بنچ",
    "bicycle": "سائیکل",
    "bird": "پرندہ",
    "both": "دونوں",
    "bridge": "پل",
    "bring": "لاؤ",
    "bulb": "بلب",
    "cartoon": "کارٹون",
    "chimpanzee": "چمپینزی",
    "color_pencils": "رنگین پنسلیں",
    "cow": "گائے",
    "crow": "کوا",
    "cupboard": "الماری",
    "deer": "ہرن",
    "dog": "کتا",
    "donttouch": "ہاتھ نہ لگاؤ",
    "door": "دروازہ",
    "elephant": "ہاتھی",
    "excuseme": "معاف کیجیے",
    "facelotion": "فیس لوشن",
    "fan": "پنکھا",
    "garden": "باغ",
    "generator": "جنریٹر",
    "goodbye": "خدا حافظ",
    "goodmorning": "صبح بخیر",
    "have_a_good_day": "اچھا دن گزاریں",
    "hello": "ہیلو",
    "ihaveacomplaint": "میری شکایت ہے",
    "left_hand": "بایاں ہاتھ",
    "lifejacket": "لائف جیکٹ",
    "mine": "میرا",
    "mobile_phone": "موبائل فون",
    "nailcutter": "ناخن تراش",
    "peacock": "مور",
    "policecar": "پولیس کار",
    "razor": "استرا",
    "s": "س",
    "shampoo": "شیمپو",
    "shower": "شاور",
    "sunglasses": "دھوپ کے چشمے",
    "thankyou": "شکریہ",
    "tissue": "ٹشو",
    "toothbrush": "ٹوتھ برش",
    "toothpaste": "ٹوتھ پیسٹ",
    "umbrella": "چھتری",
    "water": "پانی",
    "we": "ہم",
    "welldone": "شاباش",
    "you": "تم",
}


# ---------------------------------------------------------------------
# Model + encoder
# ---------------------------------------------------------------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Encoder not found: {ENCODER_PATH}")

    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()
    out_det = interpreter.get_output_details()

    # Some TFLite builds need an explicit resize to match our batch=1 window
    interpreter.resize_tensor_input(in_det[0]["index"], [1, T, F])
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()
    out_det = interpreter.get_output_details()

    encoder = joblib.load(ENCODER_PATH)
    return interpreter, in_det, out_det, encoder


# ---------------------------------------------------------------------
# Feature extraction — build the 126-D vector in the same layout as training
# ---------------------------------------------------------------------
def frame_from_mediapipe(results):
    """Return (126,) float32 vector: [lh_xyz*21][rh_xyz*21].

    Reads from MediaPipe Holistic, which assigns `left_hand_landmarks`
    and `right_hand_landmarks` anatomically (not based on image-side
    handedness). Missing hand -> zeros. This layout matches the
    training data exactly.
    """
    def flatten(landmark_list):
        if landmark_list is None:
            return np.zeros(63, dtype=np.float32)
        return np.array(
            [[p.x, p.y, p.z] for p in landmark_list.landmark],
            dtype=np.float32,
        ).reshape(-1)

    lh = flatten(results.left_hand_landmarks)
    rh = flatten(results.right_hand_landmarks)
    return np.concatenate([lh, rh])


def normalize_frame(frame):
    """Match build_word_dataset.normalize_frame exactly."""
    f = frame.reshape(2, 21, 3).copy()
    for h in range(2):
        hand = f[h]
        if not np.any(hand):
            continue
        wrist = hand[0].copy()
        hand -= wrist
        m = np.max(np.abs(hand))
        if m > 0:
            hand /= m
        f[h] = hand
    return f.reshape(126)


# ---------------------------------------------------------------------
# Motion gate — cheap "is the user actively signing?"
# ---------------------------------------------------------------------
def window_motion(window_raw):
    """Variance of wrist positions across the window, before normalization.

    Returns 0.0 when both wrists are missing; a small positive number
    when hands are present but still; larger when they are moving.
    """
    arr = np.asarray(window_raw, dtype=np.float32).reshape(len(window_raw), 2, 21, 3)
    wrists = arr[:, :, 0, :]               # (T, 2 hands, 3 coords)
    # Ignore all-zero wrist entries (missing hand) by masking
    mask = np.any(wrists != 0, axis=-1, keepdims=True).astype(np.float32)
    if mask.sum() < 5:                      # too few frames with a hand
        return 0.0
    # Variance of xyz over time, averaged across hands/coords, weighted by mask
    var = np.var(wrists * mask, axis=0).mean()
    return float(var)


# ---------------------------------------------------------------------
# Urdu rendering + TTS (lifted from psl-v1.py, simplified)
# ---------------------------------------------------------------------
def to_urdu_display(words):
    """words: list of English class names -> reshaped RTL Urdu string."""
    urdu = " ".join(PSL_WORD_TO_URDU.get(w, w) for w in words)
    if not urdu:
        return ""
    if ARABIC_SUPPORT:
        return get_display(arabic_reshaper.reshape(urdu))
    return urdu[::-1]


def put_urdu_text(img, text, position, font_size=40, color=(0, 255, 255)):
    if not text:
        return img
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = None
    for candidate in [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]:
        try:
            font = ImageFont.truetype(candidate, font_size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def speak_urdu(word_en):
    if not TTS_SUPPORT:
        return
    urdu = PSL_WORD_TO_URDU.get(word_en, word_en)

    def run():
        try:
            async def gen():
                comm = edge_tts.Communicate(urdu, VOICE, rate="-20%")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    path = fp.name
                await comm.save(path)
                return path
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            path = loop.run_until_complete(gen())
            loop.close()
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            pygame.mixer.music.unload()
            try: os.remove(path)
            except Exception: pass
        except Exception as ex:
            print(f"[TTS] {ex}")

    threading.Thread(target=run, daemon=True).start()


# ---------------------------------------------------------------------
# Prediction state machine
# ---------------------------------------------------------------------
class Committer:
    """Decides when to commit a predicted word to the sentence."""

    def __init__(self, classes):
        self.classes = list(classes)
        self.idle_indices = {
            i for i, c in enumerate(self.classes) if c in IDLE_CLASSES
        }
        self.ema = None
        self.last_commit_class = None
        self.last_commit_ts = 0.0
        self.consec_class = None
        self.consec_n = 0
        self.last_was_idle = True   # treat start-of-stream as idle

    def update(self, probs, now):
        # EMA over softmax
        if self.ema is None:
            self.ema = probs.copy()
        else:
            self.ema = EMA_ALPHA * self.ema + (1.0 - EMA_ALPHA) * probs

        top = int(np.argmax(self.ema))
        conf = float(self.ema[top])
        label = self.classes[top]
        is_idle = top in self.idle_indices

        if is_idle:
            self.last_was_idle = True
            self.consec_class = None
            self.consec_n = 0
            return label, conf, None

        # Count consecutive same-class windows
        if top == self.consec_class:
            self.consec_n += 1
        else:
            self.consec_class = top
            self.consec_n = 1

        committed = None
        cooldown_ok = (now - self.last_commit_ts) >= COMMIT_COOLDOWN_S
        boundary_ok = (
            self.last_was_idle
            or label != self.last_commit_class
            or cooldown_ok
        )

        if (
            self.consec_n >= K_CONSECUTIVE
            and conf >= COMMIT_CONF
            and boundary_ok
        ):
            committed = label
            self.last_commit_class = label
            self.last_commit_ts = now
            self.last_was_idle = False
            self.consec_n = 0

        return label, conf, committed


# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------
def run():
    print("=" * 70)
    print("Listen - PSL Word-Level Recognition")
    print("=" * 70)

    interpreter, in_det, out_det, encoder = load_model()
    classes = list(encoder.classes_)
    committer = Committer(classes)
    print(f"Loaded model, {len(classes)} classes.")

    mp_holistic = mp.solutions.holistic
    mp_hands_sol = mp.solutions.hands  # for HAND_CONNECTIONS in drawing
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    raw_buf = deque(maxlen=T)         # un-normalized, for motion gate
    norm_buf = deque(maxlen=T)        # normalized, fed to model
    sentence = []                     # list[str] of committed English classes
    top_label, top_conf = "-", 0.0
    frames_since_predict = 0
    fps_t0 = time.time()
    fps_n = 0
    fps = 0.0

    print("\nControls: 'q' quit | 'c' clear sentence | 's' speak sentence | 'space' speak last word")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # IMPORTANT: MediaPipe's handedness classifier assumes a non-mirrored
        # image. Process the raw frame for landmark extraction, draw the
        # skeleton on the un-flipped frame (so coords match), then flip the
        # whole frame for the user so the display feels like a mirror.
        h_img, w_img, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        for hlm in (results.left_hand_landmarks, results.right_hand_landmarks):
            if hlm is not None:
                mp_draw.draw_landmarks(
                    frame, hlm, mp_hands_sol.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

        frame = cv2.flip(frame, 1)

        # 1. Feature extraction from this frame
        raw_vec = frame_from_mediapipe(results)
        norm_vec = normalize_frame(raw_vec)
        raw_buf.append(raw_vec)
        norm_buf.append(norm_vec)

        # 2. Every STRIDE frames, once buffer is full, run model
        frames_since_predict += 1
        committed = None
        if len(norm_buf) == T and frames_since_predict >= STRIDE:
            frames_since_predict = 0
            motion = window_motion(list(raw_buf))
            if motion < MOTION_VAR_MIN:
                # Idle; nudge committer with a synthetic idle vote by
                # feeding zeros — model decides but we bias the gate.
                top_label, top_conf, committed = "idle", 0.0, None
                committer.last_was_idle = True
                committer.consec_n = 0
                committer.consec_class = None
            else:
                x = np.asarray(norm_buf, dtype=np.float32).reshape(1, T, F)
                interpreter.set_tensor(in_det[0]["index"], x)
                interpreter.invoke()
                probs = interpreter.get_tensor(out_det[0]["index"])[0]
                now = time.time()
                top_label, top_conf, committed = committer.update(probs, now)

        if committed:
            sentence.append(committed)
            print(f"+ {committed}  ->  {' '.join(sentence)}")
            speak_urdu(committed)

        # 3. Draw text overlay on top of the (already flipped) frame
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w_img, 260), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        color = (0, 255, 0) if top_conf > COMMIT_CONF else (0, 165, 255)
        cv2.putText(frame, f"{top_label} ({top_conf*100:.1f}%)",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)

        cv2.putText(frame, "Sentence:", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        eng = " ".join(sentence) or "(empty)"
        eng_disp = eng if len(eng) < 80 else "..." + eng[-77:]
        cv2.putText(frame, eng_disp, (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

        urdu_text = to_urdu_display(sentence)
        if urdu_text:
            cv2.putText(frame, "Urdu:", (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            frame = put_urdu_text(frame, urdu_text, (20, 200), font_size=44,
                                  color=(0, 255, 255))

        # Buffer fill + FPS indicator
        fill = min(1.0, len(norm_buf) / T)
        bw = int(fill * 200)
        cv2.rectangle(frame, (w_img - 230, 20), (w_img - 230 + bw, 35),
                      (0, 200, 0), -1)
        cv2.rectangle(frame, (w_img - 230, 20), (w_img - 30, 35),
                      (255, 255, 255), 2)
        cv2.putText(frame, f"buffer {int(fill*100)}%",
                    (w_img - 230, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)

        fps_n += 1
        if fps_n >= 10:
            fps = fps_n / (time.time() - fps_t0)
            fps_t0 = time.time(); fps_n = 0
        cv2.putText(frame, f"FPS: {fps:4.1f}",
                    (w_img - 150, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (100, 255, 100), 2)

        cv2.putText(frame,
                    "q:quit  c:clear  s:speak sentence  space:speak last",
                    (20, h_img - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1)

        cv2.imshow("Listen - PSL Words", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            sentence = []
            print("Sentence cleared.")
        elif key == ord('s'):
            if sentence:
                # Speak the whole sentence by pronouncing each word in sequence
                for w in sentence:
                    speak_urdu(w)
        elif key == ord(' '):
            if sentence:
                speak_urdu(sentence[-1])

    cap.release()
    cv2.destroyAllWindows()
    print("\nFinal sentence:", " ".join(sentence))


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] {e}")
