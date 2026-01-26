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
from PIL import Image, ImageDraw, ImageFont

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
MODEL_PATH = "models/psl/psl_landmark_classifier.tflite"
ENCODER_PATH = "src/data/psl_processed/label_encoder.pkl"
SCALER_PATH = "src/data/psl_processed/scaler.pkl"

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

# ---------------------------------------------------------------------
# 📖 Urdu Word Dictionary & Phrases for Prediction
# ---------------------------------------------------------------------
URDU_WORD_DICT = {
    # Common greetings
    "Alif Seen Lam Alif Meem": "اسلام",  # Islam
    "Seen Lam Alif Meem": "سلام",  # Salam (hello)
    "Alif Chay Dochahay Alif": "اچھا",  # Acha (good)
    "Sheen Kaaf Ray": "شکر",  # Shukr (thanks)
    
    # Common words
    "Nuun Alif Meem": "نام",  # Naam (name)
    "Pay Alif Nuun Zaey": "پانی",  # Paani (water)
    "Kaf Tay Alif Bay": "کتاب",  # Kitaab (book)
    "Gaaf Ray Meem": "گرم",  # Garm (hot)
    "Tay Nuun Dal Ray Seen Tay": "تندرست",  # Tandrust (healthy)
    "Meem Hay Bay Tay": "محبت",  # Mohabbat (love)
    "Dal Wao Seen Tay": "دوست",  # Dost (friend)
    "Khay Wao Sheen": "خوش",  # Khush (happy)
    "Bay Alif Tay": "بات",  # Baat (talk)
    "Kaaf Yay Seen Alif": "کیسا",  # Kaisa (how)
    "Kaaf Yay Seen Zaey": "کیسی",  # Kaisi (how - feminine)
    "Hay Alif Lam": "حال",  # Haal (condition)
    "Tay Meem": "تم",  # Tum (you)
    "Meem Alif Zaey Nuun": "ماں",  # Maa (mother)
    "Bay Alif Pay": "باپ",  # Baap (father)
    "Bay Hay Nuun": "بہن",  # Behan (sister)
    "Bay Hay Alif Zaey": "بھائی",  # Bhai (brother)
    "Gaaf Hay Ray": "گھر",  # Ghar (house)
    "Seen Kaaf Wao Lam": "سکول",  # School
    "Wao Kaaf Tay": "وقت",  # Waqt (time)
    "Dal Nuun": "دن",  # Din (day)
    "Ray Alif Tay": "رات",  # Raat (night)
}

# Common Urdu phrases for sentence completion
URDU_PHRASES = [
    ("Alif Seen Lam Alif Meem Alif Lam Yay Kaaf Meem", "السلام علیکم", "Assalam-o-Alaikum"),
    ("Wao Alif Lam Yay Kaaf Meem Alif Seen Lam Alif Meem", "وعلیکم السلام", "Wa-Alaikum-Salaam"),
    ("Sheen Kaaf Ray Yay Alif", "شکریہ", "Thank you"),
    ("Khay Wao Sheen Alif Meem Dal Yay Dal", "خوش آمدید", "Welcome"),
    ("Alif Chay Dochahay Alif", "اچھا", "Good/Okay"),
    ("Tay Meem Kaaf Yay Seen Alif Hay Wao", "تم کیسے ہو", "How are you (m)"),
    ("Tay Meem Kaaf Yay Seen Zaey Hay Wao", "تم کیسی ہو", "How are you (f)"),
    ("Meem Alif Zaey Nuun Alif Chay Dochahay Alif Hay Alif Zaey", "میں اچھا ہوں", "I am fine (m)"),
    ("Meem Alif Zaey Nuun Alif Chay Dochahay Zaey Hay Wao Nuun", "میں اچھی ہوں", "I am fine (f)"),
    ("Alif Lam Lam Hay Kaaf Alif Sheen Kaaf Ray", "اللہ کا شکر", "Thanks to Allah"),
]

def predict_words(current_letters):
    """
    Predict possible words based on current letter sequence.
    
    Args:
        current_letters: String of current letters (e.g., "Alif Chay")
        
    Returns:
        List of tuples: (letter_sequence, urdu_word, remaining_letters)
    """
    predictions = []
    current = current_letters.strip()
    
    if not current:
        return predictions
    
    # Find words that start with current letters
    for letter_seq, urdu_word in URDU_WORD_DICT.items():
        if letter_seq.startswith(current):
            # Calculate remaining letters needed
            remaining = letter_seq[len(current):].strip()
            predictions.append((letter_seq, urdu_word, remaining))
    
    # Sort by length (shorter completions first)
    predictions.sort(key=lambda x: len(x[2]))
    
    return predictions[:5]  # Return top 5 predictions

def suggest_phrases(current_sentence):
    """
    Suggest common phrases based on current sentence.
    
    Args:
        current_sentence: Current accumulated sentence
        
    Returns:
        List of tuples: (letter_sequence, urdu_text, english_meaning)
    """
    suggestions = []
    current = current_sentence.strip()
    
    if not current:
        # Return common greetings if nothing typed
        return URDU_PHRASES[:3]
    
    # Find phrases that start with current sentence
    for letter_seq, urdu_text, english in URDU_PHRASES:
        if letter_seq.startswith(current):
            suggestions.append((letter_seq, urdu_text, english))
    
    return suggestions[:3]  # Return top 3 suggestions


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
    
    # Join characters (left to right as typed)
    urdu_text = "".join(urdu_chars)
    
    if not urdu_text:
        return ""
    
    # Use Arabic reshaper if available for proper character connection
    if ARABIC_SUPPORT:
        # Reshape Arabic text (connects characters properly)
        reshaped_text = arabic_reshaper.reshape(urdu_text)
        # Apply bidirectional algorithm for RTL display
        display_text = get_display(reshaped_text)
        return display_text
    else:
        # Fallback: just reverse for RTL (won't connect properly)
        return urdu_text[::-1]

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
    
    # Try to use a system font that supports Arabic/Urdu
    # macOS has Arial Unicode MS which supports Urdu
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", font_size)
    except:
        try:
            # Fallback to another common font
            font = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf", font_size)
        except:
            # If no font found, use default (may not display correctly)
            font = ImageFont.load_default()
    
    # Convert BGR to RGB for PIL
    rgb_color = (color[2], color[1], color[0])
    
    # Draw text
    draw.text(position, text, font=font, fill=rgb_color)
    
    # Convert back to OpenCV BGR
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
        cv2.rectangle(overlay, (0, 0), (w, 260), (0, 0, 0), -1)
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
                stability_bar = int((stable_count / 40) * 200)
                cv2.rectangle(frame, (20, 70), (20 + stability_bar, 85),
                            (0, 255, 0), -1)
                cv2.rectangle(frame, (20, 70), (220, 85), (255, 255, 255), 2)
        else:
            # No hand detected
            cv2.putText(frame, "No hand detected", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Display accumulated sentence (letter names)
        cv2.putText(frame, "Letters:", (20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Word wrap for long sentences
        display_sentence = sentence if len(sentence) < 50 else "..." + sentence[-47:]
        cv2.putText(frame, display_sentence, (20, 155),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        
        # Display Urdu script translation
        urdu_text = convert_to_urdu_script(sentence)
        if urdu_text:
            cv2.putText(frame, "Urdu:", (20, 185),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            # Display Urdu text using PIL (supports Unicode)
            frame = put_urdu_text(frame, urdu_text, (20, 210), font_size=50, color=(0, 255, 255))
        
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
        
        # Get current word (last segment after |)
        parts = sentence.split(" | ")
        current_word = parts[-1].strip()
        
        # Word predictions
        if current_word:
            predictions = predict_words(current_word)
            if predictions:
                # Draw prediction box
                pred_y = 270
                cv2.putText(frame, "Predictions (press 1-5):", (20, pred_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
                
                for idx, (full_seq, urdu_word, remaining) in enumerate(predictions, 1):
                    pred_y += 35
                    # Show number with cv2
                    cv2.putText(frame, f"{idx}.", (30, pred_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 2)
                    
                    # Display Urdu word using PIL
                    frame = put_urdu_text(frame, urdu_word, (60, pred_y - 25), 
                                         font_size=30, color=(150, 255, 150))
                    
                    # Show remaining letters count
                    if remaining:
                        cv2.putText(frame, f"(+{len(remaining.split())})", (200, pred_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 100), 1)
        
        # Phrase suggestions
        phrase_suggestions = suggest_phrases(sentence)
        if phrase_suggestions and not current_word:
            # Draw phrase suggestion box
            phrase_y = 270
            cv2.putText(frame, "Common Phrases:", (w - 350, phrase_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 255), 2)
            
            for idx, (_, urdu_text, english) in enumerate(phrase_suggestions, 1):
                phrase_y += 25
                cv2.putText(frame, f"{english}", (w - 340, phrase_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit | 'c' to clear | 'space' for space | 1-5 for predictions",
                   (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("Listen - PSL Real-Time Recognition", frame)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            sentence = ""
            print("Sentence cleared")
        elif key == ord(' '):
            # Get the current word (last segment after |)
            parts = sentence.split(" | ")
            current_word = parts[-1].strip()
            
            if current_word:
                # Speak the word
                speak_word(current_word)
                print(f"🔊 Speaking: {current_word}")
            
            # Add space separator
            sentence += " | "
            print("Space added")
        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
            # Handle prediction selection
            pred_num = int(chr(key))
            parts = sentence.split(" | ")
            current_word = parts[-1].strip()
            
            if current_word:
                predictions = predict_words(current_word)
                if predictions and pred_num <= len(predictions):
                    # Get the selected prediction
                    selected_seq, selected_urdu, _ = predictions[pred_num - 1]
                    
                    # Replace current word with selected prediction
                    parts[-1] = selected_seq + " "
                    sentence = " | ".join(parts)
                    
                    print(f"✓ Auto-completed to: {selected_urdu}")
                    
                    # Optionally speak the completed word
                    speak_word(selected_seq)

    
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

