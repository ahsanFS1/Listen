"""
Listen - PSL Alphabet Recognition (Enhanced UI with finger-based interaction).
"""
from __future__ import annotations
import asyncio, os, tempfile, threading, time
from collections import deque
from typing import List, Optional, Tuple
import cv2, joblib, mediapipe as mp, numpy as np, tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

try:
    import psycopg2
    from dotenv import load_dotenv; load_dotenv()
    DB_SUPPORT = True
except Exception: DB_SUPPORT = False

try:
    import edge_tts, pygame; pygame.mixer.init()
    TTS_SUPPORT = True; TTS_VOICE = "ur-PK-UzmaNeural"
except Exception: TTS_SUPPORT = False; TTS_VOICE = ""

# ── Config ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "psl", "psl_landmark_classifier.tflite")
ENCODER_PATH = os.path.join(PROJECT_ROOT, "models", "psl", "label_encoder.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "psl", "scaler.pkl")

DISPLAY_WIDTH, DISPLAY_HEIGHT = 1440, 810
LEFT_PANEL_FRAC = 0.60
WINDOW_TITLE = "Listen - PSL Alphabet Recognition"
STABLE_REQUIRED = 15      # Increased to slow down locking
COOLDOWN_FRAMES = 30     # Increased to add more pause between letters
SUGGESTIONS_MAX = 5
HOVER_TRIGGER = 0.35     # Increased to make selection more intentional

# Colors (BGR)
COLOR_BG = (22, 22, 28); COLOR_PANEL = (34, 34, 44); COLOR_PANEL_LIGHT = (55, 55, 70)
COLOR_ACCENT = (240, 195, 80); COLOR_TEXT = (240, 240, 240); COLOR_TEXT_DIM = (170, 170, 180)
COLOR_OK = (80, 210, 120); COLOR_WARN = (80, 180, 240); COLOR_ERR = (80, 80, 230)

PSL_TO_URDU = {
    "Ain": "ع", "Alif": "ا", "Alifmad": "آ", "Aray": "ڑ", "Bay": "ب", "Byeh": "ے",
    "Chay": "چ", "Cyeh": "ی", "Daal": "ڈ", "Dal": "د", "Dochahay": "ھ", "Fay": "ف",
    "Gaaf": "گ", "Ghain": "غ", "Hamza": "ء", "Hay": "ح", "Jeem": "ج", "Kaf": "ک",
    "Khay": "خ", "Kiaf": "ق", "Lam": "ل", "Meem": "م", "Nuun": "ن", "Nuungh": "ں",
    "Pay": "پ", "Ray": "ر", "Say": "ث", "Seen": "س", "Sheen": "ش", "Suad": "ص",
    "Taay": "ط", "Tay": "ت", "Tuey": "ٹ", "Wao": "و", "Zaal": "ذ", "Zaey": "ی",
    "Zay": "ز", "Zuad": "ض", "Zuey": "ظ",
}

# ── Font ──
def _find(paths):
    for p in paths:
        if os.path.exists(p): return p
    return None

URDU_FONT = _find(["/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
                    "/usr/share/fonts/truetype/noto/NotoNastaliqUrdu-Regular.ttf"])
LATIN_FONT = _find(["/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
                     "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"])
_FC: dict = {}
def get_font(size, latin=False):
    k = ("L" if latin else "U", size)
    if k not in _FC:
        p = LATIN_FONT if latin else URDU_FONT
        try: _FC[k] = ImageFont.truetype(p, size) if p else ImageFont.load_default()
        except: _FC[k] = ImageFont.load_default()
    return _FC[k]

# ── TTS ──
def speak_async(text):
    if not TTS_SUPPORT or not text.strip(): return
    def _t():
        try:
            async def _r():
                c = edge_tts.Communicate(text, TTS_VOICE)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f: tmp = f.name
                await c.save(tmp); return tmp
            loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
            p = loop.run_until_complete(_r())
            pygame.mixer.music.load(p); pygame.mixer.music.play()
            while pygame.mixer.music.get_busy(): time.sleep(0.1)
            pygame.mixer.music.unload(); os.remove(p)
        except Exception as e: print(f"[TTS] {e}")
    threading.Thread(target=_t, daemon=True).start()

# ── Text Batch (single PIL round-trip per frame) ──
class _TB:
    def __init__(self): self.ops = []
    def text(self, xy, t, sz, col=COLOR_TEXT, urdu=False):
        if t: self.ops.append(("t", xy, t, sz, col, urdu))
    def rtl(self, rxy, t, sz, col=COLOR_TEXT):
        if t: self.ops.append(("r", rxy, t, sz, col))
    def wrap(self, rx, ty, t, sz, mw, lh, ml, col=COLOR_TEXT):
        if t: self.ops.append(("w", rx, ty, t, sz, mw, lh, ml, col))

    def flush(self, img):
        if not self.ops: return img
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        for op in self.ops:
            k = op[0]
            if k == "t":
                _, xy, txt, sz, col, urdu = op
                f = get_font(sz, latin=not urdu); rgb = (col[2], col[1], col[0])
                d.text(xy, txt, font=f, fill=rgb, direction="rtl" if urdu else "ltr")
            elif k == "r":
                _, rxy, txt, sz, col = op
                f = get_font(sz); rgb = (col[2], col[1], col[0])
                bb = d.textbbox((0,0), txt, font=f, direction="rtl")
                d.text((rxy[0]-(bb[2]-bb[0]), rxy[1]), txt, font=f, fill=rgb, direction="rtl")
            elif k == "w":
                _, rx, ty, txt, sz, mw, lh, ml, col = op
                f = get_font(sz); rgb = (col[2], col[1], col[0])
                words = txt.split(); lines=[]; cur=[]
                for w in words:
                    c = " ".join(cur+[w])
                    bb = d.textbbox((0,0), c, font=f, direction="rtl")
                    if (bb[2]-bb[0]) <= mw or not cur: cur.append(w)
                    else: lines.append(" ".join(cur)); cur=[w]
                if cur: lines.append(" ".join(cur))
                for i, ln in enumerate(lines[-ml:]):
                    bb = d.textbbox((0,0), ln, font=f, direction="rtl")
                    d.text((rx-(bb[2]-bb[0]), ty+i*lh), ln, font=f, fill=rgb, direction="rtl")
        self.ops.clear()
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# ── App ──
class PSLApp:
    def __init__(self):
        # Model
        self.interp = tf.lite.Interpreter(model_path=MODEL_PATH)
        self.interp.allocate_tensors()
        self.in_d = self.interp.get_input_details()
        self.out_d = self.interp.get_output_details()
        self.encoder = joblib.load(ENCODER_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        # State
        self.letters: List[str] = []        # current word's PSL labels
        self.words_urdu: List[str] = []     # confirmed Urdu words
        self.suggestions: List[str] = []
        self.last_label = ""; self.stable = 0; self.cooldown = 0
        self.confidence = 0.0
        # Finger-tip position (None when no hand)
        self.finger_tip: Optional[Tuple[int,int]] = None
        # Buttons with hover-dwell timers
        self.buttons = {n: {"rect": (0,0,0,0), "hover": 0.0}
                        for n in ("CLEAR", "SPACE", "SPEAK", "UNDO")}
        # Suggestion hover states
        self.sug_hovers = [0.0] * SUGGESTIONS_MAX
        self.sug_rects = [(0,0,0,0)] * SUGGESTIONS_MAX
        # Suggestion type tracking
        self.sug_type = "word"  # "word" or "sentence"
        # DB
        self.db = None
        if DB_SUPPORT:
            try: self.db = psycopg2.connect("postgresql://postgres:12345@localhost:5432/urdu_dict")
            except Exception as e: print(f"[DB] {e}")
        # Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ── Button actions ──
    def _do_btn(self, name):
        if name == "CLEAR":
            self.letters.clear(); self.words_urdu.clear()
        elif name == "UNDO":
            if self.letters: self.letters.pop()
            elif self.words_urdu: self.words_urdu.pop()
        elif name == "SPACE":
            if self.letters:
                w = "".join(PSL_TO_URDU.get(l,"") for l in self.letters)
                if w: self.words_urdu.append(w)
                self.letters.clear()
        elif name == "SPEAK":
            t = " ".join(self.words_urdu)
            live = "".join(PSL_TO_URDU.get(l,"") for l in self.letters)
            if live: t += (" " if t else "") + live
            speak_async(t)

    # ── Finger hover detection ──
    def _update_finger_interact(self, dt: float):
        # Top buttons
        for name, btn in self.buttons.items():
            bx, by, bw, bh = btn["rect"]
            if (self.finger_tip and
                bx <= self.finger_tip[0] <= bx+bw and
                by <= self.finger_tip[1] <= by+bh):
                btn["hover"] += dt
                if btn["hover"] >= HOVER_TRIGGER:
                    self._do_btn(name)
                    btn["hover"] = -0.8  # Longer cooldown
            else:
                btn["hover"] = max(0.0, btn["hover"] - dt*2)

        # Suggestion slots
        for i in range(len(self.suggestions)):
            if i >= len(self.sug_rects): break
            bx, by, bw, bh = self.sug_rects[i]
            if (self.finger_tip and
                bx <= self.finger_tip[0] <= bx+bw and
                by <= self.finger_tip[1] <= by+bh):
                self.sug_hovers[i] += dt
                if self.sug_hovers[i] >= HOVER_TRIGGER:
                    self._select_suggestion(i)
                    self.sug_hovers[i] = -1.0
            else:
                self.sug_hovers[i] = max(0.0, self.sug_hovers[i] - dt*2)

    # ── Suggestions ──
    def _refresh_suggestions(self):
        self.suggestions = []
        if not self.db: return
        live = "".join(PSL_TO_URDU.get(l,"") for l in self.letters)
        try:
            cur = self.db.cursor()
            if live:
                # Word completion from current letters
                self.sug_type = "word"
                cur.execute("SELECT word FROM urdu_words WHERE word LIKE %s LIMIT %s",
                            (f"{live}%", SUGGESTIONS_MAX))
            else:
                # Sentence suggestions from confirmed words
                self.sug_type = "sentence"
                ctx = " ".join(self.words_urdu)
                if ctx:
                    cur.execute(
                        "SELECT sentence FROM urdu_sentences "
                        "WHERE sentence LIKE %s OR sentence LIKE %s LIMIT %s",
                        (f"{ctx}%", f"{ctx} %", SUGGESTIONS_MAX))
                else:
                    cur.execute("SELECT sentence FROM urdu_sentences LIMIT %s",
                                (SUGGESTIONS_MAX,))
            self.suggestions = [r[0] for r in cur.fetchall()]
            cur.close()
        except Exception as e:
            print(f"[DB] {e}")
            try: self.db.rollback()
            except: pass

    def _select_suggestion(self, idx):
        if idx >= len(self.suggestions): return
        sug = self.suggestions[idx]
        if self.sug_type == "word":
            # Replace current letters with the completed word
            self.words_urdu.append(sug)
            self.letters.clear()
        else:
            # Replace entire sentence
            self.words_urdu = [sug]
            self.letters.clear()

    # ── Draw button with progress fill ──
    def _draw_btn(self, canvas, batch, name, color, bx, by, bw, bh):
        self.buttons[name]["rect"] = (bx, by, bw, bh)
        prog = max(0.0, self.buttons[name]["hover"]) / HOVER_TRIGGER
        # Background
        cv2.rectangle(canvas, (bx, by), (bx+bw, by+bh), (30, 30, 40), -1)
        # Progress fill
        if prog > 0:
            fill_w = int(bw * min(1.0, prog))
            cv2.rectangle(canvas, (bx, by), (bx+fill_w, by+bh), color, -1)
        # Border
        cv2.rectangle(canvas, (bx, by), (bx+bw, by+bh), (200, 200, 200), 1)
        # Label
        cv2.putText(canvas, name, (bx+12, by+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

    # ── Compose frame ──
    def _compose(self, frame, has_hand, fps, batch):
        canvas = np.full((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), COLOR_BG, dtype=np.uint8)
        lw = int(DISPLAY_WIDTH * LEFT_PANEL_FRAC)
        rp_x0, rp_x1 = lw, DISPLAY_WIDTH

        # Right panel bg
        cv2.rectangle(canvas, (lw, 0), (DISPLAY_WIDTH, DISPLAY_HEIGHT), COLOR_PANEL, -1)

        # Camera area
        cam_x0, cam_y0 = 20, 50
        cam_x1 = lw - 20
        cam_y1 = DISPLAY_HEIGHT - 200
        cam_w = cam_x1 - cam_x0
        cam_h = cam_y1 - cam_y0
        feed = cv2.resize(frame, (cam_w, cam_h))
        canvas[cam_y0:cam_y1, cam_x0:cam_x1] = feed
        cv2.rectangle(canvas, (cam_x0-2, cam_y0-2), (cam_x1+2, cam_y1+2), COLOR_PANEL_LIGHT, 2)
        batch.text((cam_x0, 15), "Live Camera", 20, COLOR_TEXT_DIM)

        # Guidance popup when no hand
        if not has_hand:
            ov = canvas.copy()
            px0 = cam_x0 + cam_w//2 - 180; py0 = cam_y0 + cam_h//2 - 25
            cv2.rectangle(ov, (px0, py0), (px0+360, py0+50), (0,0,0), -1)
            cv2.addWeighted(ov, 0.5, canvas, 0.5, 0, canvas)
            batch.text((px0+40, py0+12), "Position hand in frame", 22, COLOR_TEXT_DIM)

        # Virtual Buttons (top of camera)
        btn_y = cam_y0 + 15; btn_w = 100; btn_h = 42
        names_colors = [("CLEAR", COLOR_ERR), ("SPACE", COLOR_WARN),
                        ("SPEAK", COLOR_OK), ("UNDO", COLOR_WARN)]
        for i, (nm, col) in enumerate(names_colors):
            bx = cam_x0 + 15 + i * 115
            self._draw_btn(canvas, batch, nm, col, bx, btn_y, btn_w, btn_h)

        # Draw finger-tip cursor dot on camera
        if self.finger_tip:
            cv2.circle(canvas, self.finger_tip, 10, COLOR_ACCENT, -1)
            cv2.circle(canvas, self.finger_tip, 12, (255,255,255), 1)

        # Caption strip under camera
        cap_y0 = cam_y1 + 15; cap_y1 = DISPLAY_HEIGHT - 15
        cv2.rectangle(canvas, (cam_x0, cap_y0), (cam_x1, cap_y1), COLOR_PANEL, -1)
        cv2.rectangle(canvas, (cam_x0, cap_y0), (cam_x1, cap_y1), COLOR_PANEL_LIGHT, 1)
        full = " ".join(self.words_urdu)
        live_urdu = "".join(PSL_TO_URDU.get(l,"") for l in self.letters)
        if live_urdu: full += (" " if full else "") + live_urdu
        if full:
            batch.wrap(cam_x1-20, cap_y0+15, full, 30, cam_w-40, 42, 3)
        else:
            batch.text((cam_x0+20, cap_y0+25), "Your Urdu text will appear here...", 20, COLOR_TEXT_DIM)

        # ── Right Panel ──
        # Current letter + English label
        batch.text((rp_x0+30, 30), "Current Letter", 18, COLOR_TEXT_DIM)
        batch.text((rp_x0+30, 60), self.last_label if self.last_label else "—", 28, COLOR_ACCENT)
        # Urdu character (large)
        cur_u = PSL_TO_URDU.get(self.last_label, "")
        if cur_u:
            batch.rtl((rp_x1-40, 30), cur_u, 70, COLOR_ACCENT)

        # Confidence
        batch.text((rp_x0+30, 130), f"Confidence {self.confidence*100:.0f}%", 16, COLOR_TEXT_DIM)
        bar_w = rp_x1 - rp_x0 - 80
        cv2.rectangle(canvas, (rp_x0+40, 160), (rp_x0+40+bar_w, 175), (60,60,60), -1)
        conf_w = int(bar_w * self.confidence)
        conf_col = COLOR_OK if self.confidence > 0.7 else COLOR_WARN
        cv2.rectangle(canvas, (rp_x0+40, 160), (rp_x0+40+conf_w, 175), conf_col, -1)

        # Stability progress
        batch.text((rp_x0+30, 195), "Stability", 16, COLOR_TEXT_DIM)
        cv2.rectangle(canvas, (rp_x0+40, 220), (rp_x0+40+bar_w, 235), (60,60,60), -1)
        stab_w = int(bar_w * min(1.0, self.stable / STABLE_REQUIRED))
        cv2.rectangle(canvas, (rp_x0+40, 220), (rp_x0+40+stab_w, 235), COLOR_OK, -1)

        # Suggestions
        sug_label = "Word Suggestions" if self.sug_type == "word" else "Sentence Suggestions"
        batch.text((rp_x0+30, 270), sug_label, 18, COLOR_TEXT_DIM)
        if not self.suggestions:
            batch.text((rp_x0+30, 310), "Keep signing to see suggestions", 16, COLOR_TEXT_DIM)
        else:
            row_h = 60
            for i, sug in enumerate(self.suggestions[:SUGGESTIONS_MAX]):
                sy = 305 + i * row_h
                bx, by, bw, bh = rp_x0+20, sy, (rp_x1-rp_x0)-40, 50
                self.sug_rects[i] = (bx, by, bw, bh)
                
                # Visual hover feedback for suggestions
                prog = max(0.0, self.sug_hovers[i]) / HOVER_TRIGGER
                cv2.rectangle(canvas, (bx, by), (bx+bw, by+bh), COLOR_PANEL_LIGHT, -1)
                if prog > 0:
                    fill_w = int(bw * min(1.0, prog))
                    cv2.rectangle(canvas, (bx, by), (bx+fill_w, by+bh), (80, 100, 80), -1)
                
                cv2.rectangle(canvas, (bx, by), (bx+bw, by+bh), (120,120,140) if prog > 0 else (80,80,100), 1)
                batch.text((bx+10, sy+12), f"{i+1}.", 20, COLOR_ACCENT)
                batch.rtl((rp_x1-35, sy+8), sug, 24, COLOR_TEXT)

        # FPS
        batch.text((rp_x1-110, DISPLAY_HEIGHT-30), f"FPS {fps:.1f}", 14, COLOR_TEXT_DIM)

        return batch.flush(canvas)

    # ── Main loop ──
    def run(self):
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        last_sug = 0; fps_q = deque(maxlen=30); last_t = time.perf_counter()

        while True:
            t0 = time.perf_counter()
            dt = t0 - last_t; last_t = t0

            ok, frame = self.cap.read()
            if not ok: continue
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            has_hand = False
            self.finger_tip = None

            if results.multi_hand_landmarks:
                has_hand = True
                lm = results.multi_hand_landmarks[0]
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, lm, self.mp_hands.HAND_CONNECTIONS)

                # Map index finger tip to canvas coordinates
                lw = int(DISPLAY_WIDTH * LEFT_PANEL_FRAC)
                cam_x0, cam_y0 = 20, 50
                cam_x1 = lw - 20; cam_y1 = DISPLAY_HEIGHT - 200
                raw_x = lm.landmark[8].x; raw_y = lm.landmark[8].y
                # Mirror: camera is flipped, so finger at raw_x maps directly
                tip_x = int(cam_x0 + raw_x * (cam_x1 - cam_x0))
                tip_y = int(cam_y0 + raw_y * (cam_y1 - cam_y0))
                self.finger_tip = (tip_x, tip_y)

                # Inference
                if self.cooldown > 0:
                    self.cooldown -= 1
                else:
                    coords = []
                    for i in range(21):
                        coords.extend([lm.landmark[i].x, lm.landmark[i].y])
                    feat = self.scaler.transform([coords])
                    self.interp.set_tensor(self.in_d[0]['index'], feat.astype(np.float32))
                    self.interp.invoke()
                    probs = self.interp.get_tensor(self.out_d[0]['index'])[0]
                    idx = int(np.argmax(probs))
                    self.confidence = float(probs[idx])
                    label = self.encoder.classes_[idx]

                    if label == self.last_label:
                        self.stable += 1
                        if self.stable >= STABLE_REQUIRED:
                            if label != "nothing":
                                self.letters.append(label)
                                print(f"[+] Letter locked: {label} -> {PSL_TO_URDU.get(label,'?')}")
                            self.stable = 0; self.cooldown = COOLDOWN_FRAMES
                    else:
                        self.last_label = label; self.stable = 0

            # Update finger-hover buttons and suggestions
            self._update_finger_interact(dt)

            # Suggestions refresh (every 0.5s)
            if time.time() - last_sug > 0.5:
                self._refresh_suggestions(); last_sug = time.time()

            # FPS
            elapsed = max(0.001, time.perf_counter() - t0)
            fps_q.append(1.0 / elapsed)
            avg_fps = sum(fps_q) / len(fps_q)

            # Compose & show
            batch = _TB()
            canvas = self._compose(frame, has_hand, avg_fps, batch)
            cv2.imshow(WINDOW_TITLE, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == 27: break
            elif key == 13: self._do_btn("SPEAK")
            elif key == 32: self._do_btn("SPACE")
            elif key == 8:  self._do_btn("UNDO")
            elif ord('1') <= key <= ord('5'):
                self._select_suggestion(key - ord('1'))

        self.cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    PSLApp().run()
