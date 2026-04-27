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
import psycopg2
from PIL import Image, ImageDraw, ImageFont

try:
    db_conn = psycopg2.connect("postgresql://postgres:12345@localhost:5432/urdu_dict")
    print("[OK] Connected to PostgreSQL dictionary")
except Exception as e:
    print(f"[ERROR] Could not connect to PostgreSQL: {e}")
    db_conn = None

# Try to import Arabic reshaping libraries
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_SUPPORT = True
except ImportError:
    ARABIC_SUPPORT = False
    print("Warning: arabic-reshaper and python-bidi not installed.")
    print("Install with: pip install arabic-reshaper python-bidi")
    print("Urdu text may not display correctly without these libraries.")

# Try to import text-to-speech
try:
    import edge_tts
    import asyncio
    import pygame
    TTS_SUPPORT = True
    # Initialize pygame mixer for audio playback
    pygame.mixer.init()
    # Use a female Urdu voice from Microsoft Edge
    VOICE = "ur-PK-UzmaNeural"  # Pakistani Urdu female voice
except ImportError as e:
    TTS_SUPPORT = False
    print("Warning: edge-tts or pygame not installed.")
    print("Install with: pip install edge-tts pygame")
    print("Text-to-speech will not be available.")
except Exception as e:
    TTS_SUPPORT = False
    print(f"Warning: Could not initialize TTS: {e}")

# ---------------------------------------------------------------------
# ⚙️ Configuration
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "psl", "psl_landmark_classifier.tflite")
ENCODER_PATH = os.path.join(PROJECT_ROOT, "models", "psl", "label_encoder.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "psl", "scaler.pkl")

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
# 📚 PSL Letter to Urdu Character Mapping
# ---------------------------------------------------------------------
PSL_TO_URDU = {
    # Exact letter names from the PSL model
    "Ain": "ع",
    "Alif": "ا",
    "Alifmad": "آ",
    "Aray": "ڑ",
    "Bay": "ب",
    "Byeh": "ے",
    "Chay": "چ",
    "Cyeh": "ی",
    "Daal": "ڈ",
    "Dal": "د",
    "Dochahay": "ھ",
    "Fay": "ف",
    "Gaaf": "گ",
    "Ghain": "غ",
    "Hamza": "ء",
    "Hay": "ح",
    "Jeem": "ج",
    "Kaf": "ک",
    "Khay": "خ",
    "Kiaf": "ق",
    "Lam": "ل",
    "Meem": "م",
    "Nuun": "ن",
    "Nuungh": "ں",
    "Pay": "پ",
    "Ray": "ر",
    "Say": "ث",
    "Seen": "س",
    "Sheen": "ش",
    "Suad": "ص",
    "Taay": "ط",
    "Tay": "ت",
    "Tuey": "ٹ",
    "Wao": "و",
    "Zaal": "ذ",
    "Zaey": "ی",
    "Zay": "ز",
    "Zuad": "ض",
    "Zuey": "ظ",
}

URDU_TO_PSL = {v: k for k, v in PSL_TO_URDU.items()}

def urdu_to_psl_sequence(urdu_text):
    words = urdu_text.split()
    psl_words = []
    for w in words:
        psl_chars = []
        for char in w:
            psl_label = URDU_TO_PSL.get(char)
            if psl_label: psl_chars.append(psl_label)
        psl_words.append(" ".join(psl_chars))
    return " | ".join(psl_words)

# ---------------------------------------------------------------------
# 📖 Urdu Prediction from PostgreSQL
# ---------------------------------------------------------------------

def predict_words(current_letters):
    """
    Search database for words matching the currently formed Urdu prefix.
    """
    if not db_conn: return []
    current_urdu = convert_to_urdu_script(current_letters)
    if not current_urdu: return []
    
    try:
        with db_conn.cursor() as cur:
            cur.execute("SELECT word FROM urdu_words WHERE word LIKE %s LIMIT 5;", (f"{current_urdu}%",))
            results = cur.fetchall()
            
            # Log words retrieved from the DB
            words_found = [r[0] for r in results]
            if words_found:
                print(f"[DB LOG] Found words for '{current_urdu}': {words_found}")
            
        return [(urdu_to_psl_sequence(r[0]), r[0], "") for r in results]
    except Exception as e:
        print(f"DB Error in predict_words: {e}")
        db_conn.rollback()
        return []


def suggest_phrases(current_urdu_text):
    """
    Suggest common full-sentence phrases based on the live Urdu text so far.
    """
    if not db_conn: return []
    current_urdu_text = current_urdu_text.replace(" | ", " ").strip()
    
    try:
        with db_conn.cursor() as cur:
            if not current_urdu_text:
                cur.execute("SELECT sentence FROM urdu_sentences LIMIT 4;")
            else:
                cur.execute("SELECT sentence FROM urdu_sentences WHERE sentence LIKE %s LIMIT 5;", (f"{current_urdu_text}%",))
            results = cur.fetchall()
            
            # Log sentences retrieved from the DB
            sentences_found = [r[0] for r in results]
            if sentences_found:
                print(f"[DB LOG] Found sentences for '{current_urdu_text}': {sentences_found}")
            
        return [(urdu_to_psl_sequence(r[0]), r[0], r[0]) for r in results]
    except Exception as e:
        print(f"DB Error in suggest_phrases: {e}")
        db_conn.rollback()
        return []

def convert_to_urdu_script(sentence):
    """
    Convert PSL letter names to Urdu script characters with proper shaping.
    
    Args:
        sentence: String with PSL letter names (e.g., "Alif Chay Dochahay Alif ")
        
    Returns:
        String with properly shaped and connected Urdu characters (e.g., "اچھا")
    """
    # Split by spaces and filter empty strings
    letters = [l.strip() for l in sentence.split() if l.strip()]
    
    # Convert each letter name to Urdu character
    urdu_chars = []
    for letter in letters:
        if letter in PSL_TO_URDU:
            urdu_chars.append(PSL_TO_URDU[letter])
        elif letter == "|":  # Word separator
            urdu_chars.append(" ")  # Add space for word break
        else:
            urdu_chars.append(letter)  # Keep unknown as-is
    
    # Join characters
    urdu_text = "".join(urdu_chars)
    return urdu_text

def put_urdu_text(img, text, position, font_size=40, color=(0, 255, 255)):
    """
    Draw Urdu/Arabic text on OpenCV image using PIL.
    
    Args:
        img: OpenCV image (numpy array)
        text: Urdu text to display
        position: (x, y) tuple for text position
        font_size: Font size in pixels
        color: BGR color tuple
        
    Returns:
        Modified image with Urdu text
    """
    # Convert OpenCV BGR to PIL RGB
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Try Windows fonts first, then macOS fallbacks
    font_paths = [
        # Linux — available Noto Arabic/Urdu fonts
        "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoNastaliqUrdu-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
        # Windows
        "C:/Windows/Fonts/ARIALUNI.TTF",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        # macOS
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    font = None
    for fp in font_paths:
        try:
            if os.path.exists(fp):
                font = ImageFont.truetype(fp, font_size)
                break
        except:
            continue
    if font is None:
        font = ImageFont.load_default()
    
    rgb_color = (color[2], color[1], color[0])

    # Raqm handles RTL and shaping automatically if direction='rtl' is passed.
    # Manual reshaping/bidi causes issues with modern Pillow/Raqm.
    draw.text(position, text, font=font, fill=rgb_color, direction="rtl")

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

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
    STABLE_REQUIRED = 40  # number of stable frames before accepting letter
    
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
# 👍 Thumbs-Up Gesture Detection ("Word Done")
# ---------------------------------------------------------------------
def detect_thumbs_up(hand_landmarks):
    """
    Detect a thumbs-up gesture: thumb pointing up, all other fingers curled.
    Returns True if thumbs-up is detected.
    """
    lm = hand_landmarks.landmark

    # Thumb tip (4) should be ABOVE thumb MCP (2) — lower y = higher on screen
    thumb_up = lm[4].y < lm[2].y

    # All other fingers curled: tip y > PIP y (tip is lower in frame = curled)
    index_curled  = lm[8].y  > lm[6].y
    middle_curled = lm[12].y > lm[10].y
    ring_curled   = lm[16].y > lm[14].y
    pinky_curled  = lm[20].y > lm[18].y

    return thumb_up and index_curled and middle_curled and ring_curled and pinky_curled

def count_raised_fingers(hand_landmarks):
    """
    Count extended fingers using tip and PIP joint Y-coordinates.
    Handles Thumb using X-axis orientation (checks distance spread from palm).
    """
    lm = hand_landmarks.landmark
    count = 0
    
    # 4 Fingers
    if lm[8].y < lm[6].y: count += 1   # Index
    if lm[12].y < lm[10].y: count += 1 # Middle
    if lm[16].y < lm[14].y: count += 1 # Ring
    if lm[20].y < lm[18].y: count += 1 # Pinky
    
    # Thumb (approximate spread from palm center)
    dist_tip = abs(lm[4].x - lm[0].x)
    dist_mcp = abs(lm[2].x - lm[0].x)
    if dist_tip > dist_mcp + 0.02: 
        count += 1
        
    return count

def speak_word(word_letters):
    """
    Speak the Urdu word using Google Text-to-Speech.
    
    Args:
        word_letters: String of letter names (e.g., "Alif Chay Dochahay Alif")
    """
    if not TTS_SUPPORT:
        return
    
    # Convert letter names to Urdu script
    urdu_chars = []
    for letter in word_letters.strip().split():
        if letter in PSL_TO_URDU:
            urdu_chars.append(PSL_TO_URDU[letter])
        elif letter == "|":
            urdu_chars.append(" ")
    
    if not urdu_chars:
        return
    
    # Join to create Urdu word
    urdu_word = "".join(urdu_chars)
    
    # Speak in a separate thread to avoid blocking
    import threading
    import tempfile
    import os
    
    def speak():
        try:
            # Create async function to generate speech with edge-tts
            async def generate_speech():
                # Slow down speech rate by 20% for clearer pronunciation
                communicate = edge_tts.Communicate(urdu_word, VOICE, rate="-20%")
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                    temp_file = fp.name
                await communicate.save(temp_file)
                return temp_file
            
            # Run async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            temp_file = loop.run_until_complete(generate_speech())
            loop.close()
            
            # Play the audio
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            # Clean up
            pygame.mixer.music.unload()
            os.remove(temp_file)
        except Exception as e:
            print(f"TTS Error: {e}")
    
    thread = threading.Thread(target=speak, daemon=True)
    thread.start()


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
    urdu_parts = []              # confirmed Urdu word for each completed slot
    last_label = None
    stable_count = 0
    thumbs_up_count = 0          # frames thumbs-up held
    THUMBS_UP_REQUIRED = 30      # ~1 sec at 30fps to confirm gesture
    thumbs_up_triggered = False  # prevent repeat trigger while held
    
    # Selection Finger Tracker
    last_finger_count = 0
    finger_stable_count = 0
    FINGER_STABLE_REQUIRED = 15  # Half sec hold to select option

    fps_time = time.time()
    frame_count = 0
    
    while True:
        synthetic_key = None
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)  # Mirror the frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        h, w, _ = frame.shape
        
        # Draw semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 260), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Spatial interaction mode! Right side of frame = Menu Selection. Left/Center = Spelling.
                is_selection_mode = hand_landmarks.landmark[0].x > 0.55
                
                # Draw hand landmarks with style
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Draw Hand Mode Tracker physically on the wrist
                wrist_x = int(hand_landmarks.landmark[0].x * w)
                wrist_y = int(hand_landmarks.landmark[0].y * h)
                mode_text = "MODE: SELECTION" if is_selection_mode else "MODE: SPELLING"
                mode_color = (255, 255, 0) if is_selection_mode else (0, 255, 0)
                cv2.putText(frame, mode_text, (wrist_x - 50, wrist_y + 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
                
                if not is_selection_mode:
                    # ─── SPELLING MODE ───
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y])

                    # Thumbs-Up gesture = Word Done
                    if detect_thumbs_up(hand_landmarks):
                        thumbs_up_count += 1
                        bar_len = int((thumbs_up_count / THUMBS_UP_REQUIRED) * 200)
                        cv2.rectangle(frame, (w - 230, 10), (w - 230 + bar_len, 30), (0, 215, 255), -1)
                        cv2.rectangle(frame, (w - 230, 10), (w - 30, 30), (255, 255, 255), 2)
                        cv2.putText(frame, "👍 Word Done", (w - 230, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)

                        if thumbs_up_count >= THUMBS_UP_REQUIRED and not thumbs_up_triggered:
                            parts = sentence.split(" | ")
                            current_word = parts[-1].strip()
                            if current_word:
                                speak_word(current_word)
                                print(f"👍 Thumbs-up: word done → {current_word}")
                                urdu_parts.append(convert_to_urdu_script(current_word))
                            sentence += " | "
                            thumbs_up_triggered = True
                            thumbs_up_count = 0
                    else:
                        thumbs_up_count = 0
                        thumbs_up_triggered = False

                    # Normal sign prediction
                    label, conf = predict_sign(landmarks)
                    last_label, stable_count, sentence = update_text(
                        label, conf, last_label, stable_count, sentence
                    )
                    
                    # Display prediction
                    color = (0, 255, 0) if conf > 0.85 else (0, 165, 255)
                    cv2.putText(frame, f"{label} ({conf*100:.1f}%)", (20, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
                    
                    # Stability indicator
                    stability_bar = int((stable_count / 40) * 200)
                    cv2.rectangle(frame, (20, 70), (20 + stability_bar, 85), (0, 255, 0), -1)
                    cv2.rectangle(frame, (20, 70), (220, 85), (255, 255, 255), 2)
                
                else:
                    # ─── MENU SELECTION MODE ───
                    fc = count_raised_fingers(hand_landmarks)
                    
                    # Ensure fingers match menu options (1-5)
                    if fc == last_finger_count and fc in [1, 2, 3, 4, 5]:
                        finger_stable_count += 1
                    else:
                        finger_stable_count = 0
                        last_finger_count = fc
                    
                    # Draw visual counter for user feedback
                    cv2.putText(frame, f"Selecting: {fc}", (w - 300, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)
                    
                    if finger_stable_count > 0 and fc > 0:
                        bar_len = int((finger_stable_count / FINGER_STABLE_REQUIRED) * 150)
                        cv2.rectangle(frame, (w - 300, 95), (w - 300 + bar_len, 105), (255, 255, 0), -1)
                    
                    if finger_stable_count >= FINGER_STABLE_REQUIRED:
                        synthetic_key = ord(str(fc))
                        finger_stable_count = -15  # Cooldown
        else:
            # No hand detected
            cv2.putText(frame, "No hand detected", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        
        # ── Get current word buffer (needed for display + predictions) ──
        parts = sentence.split(" | ")
        current_word = parts[-1].strip()

        # Display accumulated sentence (letter names)
        cv2.putText(frame, "Letters:", (20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Word wrap for long sentences
        display_sentence = sentence if len(sentence) < 50 else "..." + sentence[-47:]
        cv2.putText(frame, display_sentence, (20, 155),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        
        # Display Urdu script — confirmed parts + live current buffer
        live_urdu = convert_to_urdu_script(current_word) if current_word else ""
        if urdu_parts:
            full_urdu = " ".join(urdu_parts) + (" " + live_urdu if live_urdu else "")
        else:
            full_urdu = live_urdu
        if full_urdu:
            cv2.putText(frame, "Urdu:", (20, 185),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            frame = put_urdu_text(frame, full_urdu, (20, 210), font_size=50, color=(0, 255, 255))
        
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
        
        predictions = predict_words(current_word) if current_word else []

        # sync urdu_parts length with completed word slots
        completed_slots = len(parts) - 1
        while len(urdu_parts) < completed_slots:
            urdu_parts.append("")
        while len(urdu_parts) > completed_slots:
            urdu_parts.pop()

        phrase_suggestions = suggest_phrases(full_urdu)

        # ── Right-side prediction panel ──
        panel_w = 650
        panel_x = w - panel_w - 10
        panel_top = 270
        panel_items = predictions if predictions else []
        phrases_to_show = phrase_suggestions if phrase_suggestions else []
        
        # Calculate dynamic height
        needed_h = 40
        if panel_items:
            needed_h += len(panel_items) * 52 + 10
        if phrases_to_show:
            needed_h += len(phrases_to_show) * 40 + 20
        panel_h = max(160, needed_h)
        
        panel_overlay = frame.copy()
        cv2.rectangle(panel_overlay, (panel_x - 10, panel_top),
                      (w - 5, panel_top + panel_h), (20, 20, 50), -1)
        frame = cv2.addWeighted(panel_overlay, 0.72, frame, 0.28, 0)
        cv2.rectangle(frame, (panel_x - 10, panel_top),
                      (w - 5, panel_top + panel_h), (100, 150, 255), 1)

        current_y = panel_top + 22

        if predictions:
            cv2.putText(frame, "Word Suggestions  (1-5)",
                        (panel_x - 8, current_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 255), 1)
            current_y += 18
            for idx, (full_seq, urdu_word, remaining) in enumerate(predictions, 1):
                row_color = (60, 60, 100)
                cv2.rectangle(frame, (panel_x - 8, current_y + 4),
                              (w - 8, current_y + 46), row_color, -1)
                cv2.putText(frame, f"[{idx}]",
                            (panel_x - 4, current_y + 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 80), 2)
                frame = put_urdu_text(frame, urdu_word,
                                      (panel_x + 40, current_y + 4),
                                      font_size=34, color=(80, 255, 160))
                if remaining:
                    rem_count = len(remaining.split())
                    cv2.putText(frame, f"+{rem_count} signs",
                                (panel_x + 400, current_y + 36),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 200, 160), 1)
                current_y += 52

        if phrases_to_show:
            if predictions:
                current_y += 10
            title = "Common Phrases (Press 1-5)" if not current_word else "Suggested Phrases:"
            cv2.putText(frame, title,
                        (panel_x - 8, current_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 255), 1)
            current_y += 8
            for idx, (_, urdu_text, english) in enumerate(phrases_to_show, 1):
                prefix = f"[{idx}]" if not current_word else "-"
                cv2.putText(frame, prefix,
                            (panel_x - 4, current_y + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 180, 255), 1)
                frame = put_urdu_text(frame, english,
                                      (panel_x + 35, current_y + 2),
                                      font_size=24, color=(255, 200, 255))
                current_y += 40

        if not predictions and not phrases_to_show:
            cv2.putText(frame, "No matches yet...",
                        (panel_x - 4, panel_top + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 180), 1)
            cv2.putText(frame, "Keep signing letters",
                        (panel_x - 4, panel_top + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 160), 1)

        # Instructions bar
        cv2.putText(frame,
                    "Bksp=undo | Enter=speak all | Space=next word | c=clear | q=quit",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        
        cv2.imshow("Listen - PSL Real-Time Recognition", frame)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        
        # Override with synthetic key if finger selection triggered
        if synthetic_key:
            key = synthetic_key
            print(f"[UI] Triggered selection option {chr(synthetic_key)} via gesture")
            
        if key == ord('q'):
            break
        elif key == ord('c'):
            sentence = ""
            urdu_parts = []
            print("Sentence cleared")
        elif key == 8 or key == 127:  # Backspace
            parts = sentence.split(" | ")
            if parts[-1] == "":
                # We are at a word boundary (sentence ends with " | ")
                if len(parts) > 1:
                    parts.pop()
                    sentence = " | ".join(parts)
                    if urdu_parts:
                        urdu_parts.pop()
                    print("Removed word separator")
            else:
                # Remove last letter from current word
                cw = parts[-1].rstrip()
                idx = cw.rfind(" ")
                if idx != -1:
                    parts[-1] = cw[:idx] + " "
                else:
                    parts[-1] = ""
                sentence = " | ".join(parts)
                print("Removed last letter")
        elif key == 13:  # Enter — speak the full confirmed + live Urdu
            live_urdu = convert_to_urdu_script(current_word) if current_word else ""
            full_for_speech = " ".join(urdu_parts) + (" " + live_urdu if live_urdu else "")
            if full_for_speech.strip():
                import threading, asyncio, tempfile
                def _speak_urdu_direct(text):
                    async def _gen():
                        communicate = edge_tts.Communicate(text, VOICE, rate="-20%")
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                            tmp = fp.name
                        await communicate.save(tmp)
                        return tmp
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        tmp = loop.run_until_complete(_gen())
                        loop.close()
                        pygame.mixer.music.load(tmp)
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            pygame.time.Clock().tick(10)
                        pygame.mixer.music.unload()
                        os.remove(tmp)
                    except Exception as e:
                        print(f"TTS Error: {e}")
                if TTS_SUPPORT:
                    threading.Thread(target=_speak_urdu_direct,
                                     args=(full_for_speech,), daemon=True).start()
                print(f"🔊 Speaking: {full_for_speech}")
        elif key == ord(' '):
            parts = sentence.split(" | ")
            current_word = parts[-1].strip()
            if current_word:
                speak_word(current_word)
                urdu_parts.append(convert_to_urdu_script(current_word))
                print(f"🔊 Speaking: {current_word}")
            sentence += " | "
            print("Space added")
        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
            pred_num = int(chr(key))
            parts = sentence.split(" | ")
            current_word = parts[-1].strip()
            
            if current_word:
                # Word selection
                preds_now = predict_words(current_word)
                if preds_now and pred_num <= len(preds_now):
                    selected_seq, selected_urdu, _ = preds_now[pred_num - 1]
                    # Speak the selected Urdu word directly (correct pronunciation)
                    import threading, asyncio, tempfile
                    def _speak_selected(text):
                        async def _gen():
                            communicate = edge_tts.Communicate(text, VOICE, rate="-20%")
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                                tmp = fp.name
                            await communicate.save(tmp)
                            return tmp
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            tmp = loop.run_until_complete(_gen())
                            loop.close()
                            pygame.mixer.music.load(tmp)
                            pygame.mixer.music.play()
                            while pygame.mixer.music.get_busy():
                                pygame.time.Clock().tick(10)
                            pygame.mixer.music.unload()
                            os.remove(tmp)
                        except Exception as e:
                            print(f"TTS Error: {e}")
                    if TTS_SUPPORT:
                        threading.Thread(target=_speak_selected,
                                         args=(selected_urdu,), daemon=True).start()
                    print(f"✓ Selected: {selected_urdu}")
                    # Store the exact Urdu word in confirmed parts
                    urdu_parts.append(selected_urdu)
                    # Update PSL sentence buffer
                    parts[-1] = selected_seq
                    sentence = " | ".join(parts) + " | "
                    stable_count = 0
            else:
                # Phrase Selection (When current_word is empty)
                phrase_suggs = suggest_phrases(" ".join(urdu_parts))
                if phrase_suggs and pred_num <= len(phrase_suggs):
                    selected_seq, selected_urdu, translated = phrase_suggs[pred_num - 1]
                    import threading, asyncio, tempfile
                    def _speak_selected(text):
                        async def _gen():
                            communicate = edge_tts.Communicate(text, VOICE, rate="-20%")
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                                tmp = fp.name
                            await communicate.save(tmp)
                            return tmp
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            tmp = loop.run_until_complete(_gen())
                            loop.close()
                            pygame.mixer.music.load(tmp)
                            pygame.mixer.music.play()
                            while pygame.mixer.music.get_busy():
                                pygame.time.Clock().tick(10)
                            pygame.mixer.music.unload()
                            os.remove(tmp)
                        except Exception as e:
                            print(f"TTS Error: {e}")
                    if TTS_SUPPORT:
                        threading.Thread(target=_speak_selected,
                                         args=(selected_urdu,), daemon=True).start()
                    print(f"✓ Phrase Selected: {selected_urdu}")
                    
                    # Update application state to reflect the whole phrase at once!
                    sentence = selected_seq + " | "
                    urdu_parts = selected_urdu.split()
                    stable_count = 0

    
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

