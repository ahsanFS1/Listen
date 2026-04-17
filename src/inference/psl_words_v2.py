"""
Listen - PSL Word-Level Recognition (Enhanced UI, Threaded Inference, FSM).

Drop-in replacement for psl_words_v1.py. Preserves these contracts:
    * MediaPipe Holistic configuration (model_complexity=1, det/tracking=0.5).
    * Model input shape (1, T=60, F=126) float32, same per-frame ordering
      (lh_xyz[21] ++ rh_xyz[21]) and the same normalize_frame transform.
    * Model output decoding: argmax of raw probability vector -> class index
      -> encoder.classes_[index].

Layout revision (post-feedback):
    * Word history is rendered as a flowing wrapped caption beneath the
      camera feed (no "selection" look; plain text that wraps to new lines).
    * The "Position your hands in the frame" guidance is now a small, light
      centre popup on the camera feed that auto-dismisses the moment any
      hand is detected.
    * The right panel shows sentence Suggestions where history used to sit.
    * The confidence-threshold slider stays in the right panel (mid-height)
      and nothing confidence-related sits at the bottom of the screen.
"""

from __future__ import annotations

import asyncio
import functools
import os
import queue
import sys
import tempfile
import textwrap
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Deque, List, Optional, Tuple

import cv2
import joblib
import mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

try:
    import psycopg2
    from dotenv import load_dotenv
    load_dotenv()
    DB_SUPPORT = True
except Exception:
    DB_SUPPORT = False

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_SUPPORT = True
except ImportError:
    ARABIC_SUPPORT = False

try:
    import edge_tts
    import pygame
    pygame.mixer.init()
    TTS_SUPPORT = True
    TTS_VOICE = "ur-PK-UzmaNeural"
except Exception:
    TTS_SUPPORT = False
    TTS_VOICE = ""

# =====================================================================
# Named constants
# =====================================================================

# --- model / feature contract (DO NOT CHANGE) --------------------------
T_WINDOW: int = 60
F_DIM: int = 126
HAND_COUNT: int = 2
LANDMARKS_PER_HAND: int = 21
COORDS_PER_LANDMARK: int = 3

# --- capture / display -------------------------------------------------
CAM_WIDTH: int = 1280
CAM_HEIGHT: int = 720
PROC_WIDTH: int = 640
PROC_HEIGHT: int = 360
DISPLAY_WIDTH: int = 1440
DISPLAY_HEIGHT: int = 810
LEFT_PANEL_FRAC: float = 0.60
WINDOW_TITLE: str = "Listen - PSL (Enhanced)"

# --- FSM timings / thresholds -----------------------------------------
STRIDE_FRAMES: int = 3
COMMIT_CONF_DEFAULT: float = 0.70
CONF_MIN: float = 0.40
CONF_MAX: float = 0.95
COOLDOWN_SECONDS: float = 0.8
UNDO_HOLD_SECONDS: float = 2.0
MOTION_VAR_MIN: float = 1e-4
IDLE_CLASSES = {"nothing", "test_word"}

# --- graceful-degradation prompts -------------------------------------
CONF_ROLL_WINDOW: int = 15
LOW_CONF_STREAK_THRESHOLD: int = 10
LOW_CONF_MEAN: float = 0.40
NO_HAND_PROMPT_SECONDS: float = 2.0
PROMPT_DISMISS_SECONDS: float = 3.0

# --- UI / rendering ----------------------------------------------------
WORD_HISTORY_MAX: int = 50
EMA_DISPLAY_ALPHA: float = 0.30
EMA_INFER_ALPHA: float = 0.60
SILHOUETTE_ALPHA: float = 0.28
PANEL_ALPHA: float = 0.82
CAPTION_FONT_SIZE: int = 30
CAPTION_LINE_HEIGHT: int = 44
CAPTION_MAX_LINES: int = 3
SUGGESTIONS_MAX: int = 5

# --- palette (BGR) -----------------------------------------------------
COLOR_BG = (22, 22, 28)
COLOR_PANEL = (34, 34, 44)
COLOR_PANEL_LIGHT = (55, 55, 70)
COLOR_ACCENT = (240, 195, 80)
COLOR_TEXT = (240, 240, 240)
COLOR_TEXT_DIM = (170, 170, 180)
COLOR_OK = (80, 210, 120)
COLOR_WARN = (80, 180, 240)
COLOR_ERR = (80, 80, 230)

STATE_COLORS = {
    "IDLE":       (90, 200, 120),
    "SIGNING":    (60, 210, 230),
    "PREDICTING": (60, 140, 240),
    "COMMITTED":  (230, 150, 60),
    "COOLDOWN":   (80, 80, 230),
}

# --- paths -------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(_HERE)),
    "models", "psl_words", "psl_word_classifier.tflite",
)
ENCODER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(_HERE)),
    "models", "psl_words", "label_encoder.pkl",
)
SESSION_OUTPUT_DIR = os.getcwd()

# --- PSL word -> Urdu mapping -----------------------------------------
PSL_WORD_TO_URDU = {
    "absolutely": "بالکل", "aircrash": "ہوائی حادثہ", "airplane": "ہوائی جہاز",
    "all": "سب", "also": "بھی", "arrival": "آمد",
    "assalam-o-alaikum": "السلام علیکم", "atm": "اے ٹی ایم", "bald": "گنجا",
    "beach": "ساحل", "beak": "چونچ", "bear": "ریچھ", "beard": "داڑھی",
    "bed": "بستر", "bench": "بنچ", "bicycle": "سائیکل", "bird": "پرندہ",
    "both": "دونوں", "bridge": "پل", "bring": "لاؤ", "bulb": "بلب",
    "cartoon": "کارٹون", "chimpanzee": "چمپینزی", "color_pencils": "رنگین پنسلیں",
    "cow": "گائے", "crow": "کوا", "cupboard": "الماری", "deer": "ہرن",
    "dog": "کتا", "donttouch": "ہاتھ نہ لگاؤ", "door": "دروازہ",
    "elephant": "ہاتھی", "excuseme": "معاف کیجیے", "facelotion": "فیس لوشن",
    "fan": "پنکھا", "garden": "باغ", "generator": "جنریٹر",
    "goodbye": "خدا حافظ", "goodmorning": "صبح بخیر",
    "have_a_good_day": "اچھا دن گزاریں", "hello": "ہیلو",
    "ihaveacomplaint": "میری شکایت ہے", "left_hand": "بایاں ہاتھ",
    "lifejacket": "لائف جیکٹ", "mine": "میرا", "mobile_phone": "موبائل فون",
    "nailcutter": "ناخن تراش", "peacock": "مور", "policecar": "پولیس کار",
    "razor": "استرا", "s": "س", "shampoo": "شیمپو", "shower": "شاور",
    "sunglasses": "دھوپ کے چشمے", "thankyou": "شکریہ", "tissue": "ٹشو",
    "toothbrush": "ٹوتھ برش", "toothpaste": "ٹوتھ پیسٹ",
    "umbrella": "چھتری", "water": "پانی", "we": "ہم",
    "welldone": "شاباش", "you": "تم",
}


# =====================================================================
# Font discovery
# =====================================================================
def _find_urdu_font() -> Optional[str]:
    """Return path to a font file that renders Urdu/Arabic glyphs, or None."""
    candidates = [
        "C:/Windows/Fonts/NotoNastaliqUrdu-Regular.ttf",
        "C:/Windows/Fonts/ARIALUNI.TTF",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/noto/NotoNastaliqUrdu-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/GeezaPro.ttc",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


URDU_FONT_PATH = _find_urdu_font()
_FONT_CACHE: dict = {}


def get_font(size: int) -> ImageFont.FreeTypeFont:
    """Return a cached truetype font at the requested size, or a default."""
    key = ("urdu", size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    try:
        if URDU_FONT_PATH is not None:
            f = ImageFont.truetype(URDU_FONT_PATH, size)
        else:
            f = ImageFont.load_default()
    except Exception:
        f = ImageFont.load_default()
    _FONT_CACHE[key] = f
    return f


@functools.lru_cache(maxsize=2048)
def shape_urdu(text: str) -> str:
    """Reshape + bidi an Urdu string so it renders correctly in Pillow.

    Cached: the same sentence is reshaped every frame; the cache makes
    subsequent frames ~free.
    """
    if not text:
        return ""
    if ARABIC_SUPPORT:
        try:
            return get_display(arabic_reshaper.reshape(text))
        except Exception:
            return text
    return text


# Reusable offscreen PIL draw context for text measurement. Avoids
# allocating + converting the full canvas just to measure glyph widths.
_MEASURE_IMG = Image.new("RGB", (4, 4))
_MEASURE_DRAW = ImageDraw.Draw(_MEASURE_IMG)


# =====================================================================
# FSM
# =====================================================================
class SignState(Enum):
    """Finite-state-machine states for the sign-recognition pipeline."""
    IDLE = auto()
    SIGNING = auto()
    PREDICTING = auto()
    COMMITTED = auto()
    COOLDOWN = auto()


STATE_LABEL = {
    SignState.IDLE: "IDLE",
    SignState.SIGNING: "SIGNING",
    SignState.PREDICTING: "PREDICTING",
    SignState.COMMITTED: "COMMITTED",
    SignState.COOLDOWN: "COOLDOWN",
}


# =====================================================================
# TTS (non-blocking)
# =====================================================================
def speak_text_async(text: str) -> None:
    """Speak an arbitrary Urdu string in a daemon thread. No-op if no TTS."""
    if not TTS_SUPPORT or not text:
        return

    def _worker() -> None:
        """Worker that synthesises mp3 via edge-tts and plays via pygame."""
        try:
            async def _gen() -> str:
                comm = edge_tts.Communicate(text, TTS_VOICE, rate="-20%")
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp3",
                ) as fp:
                    path = fp.name
                await comm.save(path)
                return path

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                path = loop.run_until_complete(_gen())
            finally:
                loop.close()
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            pygame.mixer.music.unload()
            try:
                os.remove(path)
            except OSError:
                pass
        except Exception as exc:
            print(f"[TTS] {exc}")

    threading.Thread(target=_worker, daemon=True).start()


def speak_word_async(english_word: str) -> None:
    """Translate an English class name to Urdu and speak it."""
    if not english_word:
        return
    text = PSL_WORD_TO_URDU.get(english_word, english_word)
    speak_text_async(text)


# =====================================================================
# Hands → Holistic adapter (accuracy-test swap)
# =====================================================================
# The model was trained on data extracted with MediaPipe Holistic, which
# returns `left_hand_landmarks` / `right_hand_landmarks` keyed to the
# signer's ANATOMICAL sides. MediaPipe Hands returns an unordered list
# plus a "Left"/"Right" handedness label from its classifier, which is
# trained to expect selfie-FLIPPED input. We feed the raw (non-flipped)
# processing frame, so the classifier's label is inverted relative to
# anatomy — a "Right" label means the anatomical left hand. Flip
# `HANDS_INVERT_HANDEDNESS` to False if you pre-flip the frame.
HANDS_INVERT_HANDEDNESS: bool = True


class _HandsToHolistic:
    """Expose mp.solutions.hands results as Holistic-style attributes."""

    __slots__ = ("left_hand_landmarks", "right_hand_landmarks")

    def __init__(self, hands_results) -> None:
        self.left_hand_landmarks = None
        self.right_hand_landmarks = None
        mhl = getattr(hands_results, "multi_hand_landmarks", None)
        if not mhl:
            return
        mhd = getattr(hands_results, "multi_handedness", None) or []
        for lm, hd in zip(mhl, mhd):
            label = hd.classification[0].label
            if HANDS_INVERT_HANDEDNESS:
                is_anatomical_left = (label == "Right")
            else:
                is_anatomical_left = (label == "Left")
            if is_anatomical_left:
                if self.left_hand_landmarks is None:
                    self.left_hand_landmarks = lm
            else:
                if self.right_hand_landmarks is None:
                    self.right_hand_landmarks = lm


# =====================================================================
# Fast hand-landmark drawer (replaces mp.solutions.drawing_utils)
# =====================================================================
# mp_drawing_utils.draw_landmarks does 42 individual cv2.circle calls
# and builds python lists on every call. We batch into a single
# cv2.polylines for the bones and cheap circles for the joints — 3-5ms
# saved per frame on a 640×360 canvas.
_HAND_BONES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm
)
_BONE_COLOR: Tuple[int, int, int] = (255, 255, 255)
_POINT_COLOR: Tuple[int, int, int] = (0, 200, 255)


def draw_hand_fast(img: np.ndarray, landmarks) -> None:
    """Render one hand's 21 landmarks into `img` in-place."""
    h, w = img.shape[:2]
    lm = landmarks.landmark
    pts = np.empty((21, 2), dtype=np.int32)
    for i in range(21):
        p = lm[i]
        pts[i, 0] = int(p.x * w)
        pts[i, 1] = int(p.y * h)
    segs = [pts[[a, b]].reshape(-1, 1, 2) for a, b in _HAND_BONES]
    cv2.polylines(img, segs, False, _BONE_COLOR, 2, cv2.LINE_AA)
    for i in range(21):
        cv2.circle(img, (int(pts[i, 0]), int(pts[i, 1])), 3,
                   _POINT_COLOR, -1, cv2.LINE_AA)


# =====================================================================
# Landmark helpers (PROTECTED — identical math to v1)
# =====================================================================
def frame_from_mediapipe(results) -> np.ndarray:
    """Flatten holistic results into the 126-D per-frame vector."""
    def _flatten(landmark_list) -> np.ndarray:
        if landmark_list is None:
            return np.zeros(
                LANDMARKS_PER_HAND * COORDS_PER_LANDMARK, dtype=np.float32,
            )
        return np.array(
            [[p.x, p.y, p.z] for p in landmark_list.landmark],
            dtype=np.float32,
        ).reshape(-1)

    lh = _flatten(results.left_hand_landmarks)
    rh = _flatten(results.right_hand_landmarks)
    return np.concatenate([lh, rh])


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Per-hand wrist-centre + max-abs normalize, identical to training."""
    f = frame.reshape(
        HAND_COUNT, LANDMARKS_PER_HAND, COORDS_PER_LANDMARK,
    ).copy()
    for h in range(HAND_COUNT):
        hand = f[h]
        if not np.any(hand):
            continue
        wrist = hand[0].copy()
        hand -= wrist
        m = np.max(np.abs(hand))
        if m > 0:
            hand /= m
        f[h] = hand
    return f.reshape(F_DIM)


# =====================================================================
# Model loading
# =====================================================================
def load_model() -> Tuple[tf.lite.Interpreter, list, list, object]:
    """Load and allocate the TFLite interpreter and the label encoder."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Encoder not found: {ENCODER_PATH}")
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()
    interpreter.resize_tensor_input(in_det[0]["index"], [1, T_WINDOW, F_DIM])
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()
    out_det = interpreter.get_output_details()
    encoder = joblib.load(ENCODER_PATH)
    return interpreter, in_det, out_det, encoder


# =====================================================================
# Pillow drawing helpers
# =====================================================================
def draw_text(img: np.ndarray, xy: Tuple[int, int], text: str, size: int,
              color: Tuple[int, int, int] = COLOR_TEXT,
              shape: bool = False) -> np.ndarray:
    """Draw Latin or Urdu text on a BGR OpenCV frame using Pillow."""
    if not text:
        return img
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    if shape:
        text = shape_urdu(text)
    rgb = (color[2], color[1], color[0])
    draw.text(xy, text, font=get_font(size), fill=rgb)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _measure_text_width(draw: ImageDraw.ImageDraw, text: str,
                        font: ImageFont.FreeTypeFont) -> int:
    """Return the pixel width of `text` when rendered in `font`."""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]
    except AttributeError:
        w, _ = draw.textsize(text, font=font)
        return w


def draw_rtl_text(img: np.ndarray, right_xy: Tuple[int, int], text: str,
                  size: int,
                  color: Tuple[int, int, int] = COLOR_TEXT) -> np.ndarray:
    """Draw Urdu text right-anchored at `right_xy`."""
    if not text:
        return img
    shaped = shape_urdu(text)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = get_font(size)
    w = _measure_text_width(draw, shaped, font)
    x = right_xy[0] - w
    y = right_xy[1]
    rgb = (color[2], color[1], color[0])
    draw.text((x, y), shaped, font=font, fill=rgb)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def draw_wrapped_rtl(img: np.ndarray, right_x: int, top_y: int,
                     text: str, size: int, max_width: int,
                     line_height: int, max_lines: int,
                     color: Tuple[int, int, int] = COLOR_TEXT) -> np.ndarray:
    """Draw an Urdu paragraph wrapped to `max_width`, right-aligned RTL.

    Word-wrap happens on whitespace. Lines that overflow `max_lines` are
    dropped (oldest first) so the most recent words always remain visible.
    """
    if not text:
        return img
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = get_font(size)
    words = text.split()
    # Build raw (unshaped) lines by measuring shaped widths so wrap is exact.
    lines: List[str] = []
    cur: List[str] = []
    for w in words:
        candidate = " ".join(cur + [w])
        shaped = shape_urdu(candidate)
        if _measure_text_width(draw, shaped, font) <= max_width or not cur:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    # Keep only the last `max_lines` lines so the newest content stays.
    lines = lines[-max_lines:]
    rgb = (color[2], color[1], color[0])
    for i, raw_line in enumerate(lines):
        shaped = shape_urdu(raw_line)
        w = _measure_text_width(draw, shaped, font)
        draw.text((right_x - w, top_y + i * line_height),
                  shaped, font=font, fill=rgb)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


class _TextBatch:
    """Queue text draws and apply them in a single PIL round-trip.

    Each of ``draw_text`` / ``draw_rtl_text`` / ``draw_wrapped_rtl`` on
    its own does a full BGR->PIL->BGR conversion of the canvas. The
    main UI fires ~20 such calls per frame, which at 1440x810 costs
    more than the MediaPipe + inference pipeline combined. Batching
    them into one conversion per frame is the single biggest win.
    """

    __slots__ = ("ops",)

    def __init__(self) -> None:
        self.ops: list = []

    def draw(self, xy: Tuple[int, int], text: str, size: int,
             color: Tuple[int, int, int] = COLOR_TEXT,
             shape: bool = False) -> None:
        """Equivalent to module-level ``draw_text`` but deferred."""
        if text:
            self.ops.append(("d", xy, text, size, color, shape))

    def rtl(self, right_xy: Tuple[int, int], text: str, size: int,
            color: Tuple[int, int, int] = COLOR_TEXT) -> None:
        """Equivalent to module-level ``draw_rtl_text`` but deferred."""
        if text:
            self.ops.append(("r", right_xy, text, size, color))

    def wrap(self, right_x: int, top_y: int, text: str, size: int,
             max_width: int, line_height: int, max_lines: int,
             color: Tuple[int, int, int] = COLOR_TEXT) -> None:
        """Equivalent to module-level ``draw_wrapped_rtl`` but deferred."""
        if text:
            self.ops.append(("w", right_x, top_y, text, size,
                             max_width, line_height, max_lines, color))

    def flush(self, img: np.ndarray) -> np.ndarray:
        """Apply all queued ops in a single PIL pass and return the image."""
        if not self.ops:
            return img
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        for op in self.ops:
            kind = op[0]
            if kind == "d":
                _, xy, text, size, color, shape = op
                t = shape_urdu(text) if shape else text
                rgb = (color[2], color[1], color[0])
                draw.text(xy, t, font=get_font(size), fill=rgb)
            elif kind == "r":
                _, right_xy, text, size, color = op
                shaped = shape_urdu(text)
                font = get_font(size)
                w = _measure_text_width(draw, shaped, font)
                rgb = (color[2], color[1], color[0])
                draw.text((right_xy[0] - w, right_xy[1]), shaped,
                          font=font, fill=rgb)
            else:  # "w"
                (_, right_x, top_y, text, size, max_width,
                 line_height, max_lines, color) = op
                font = get_font(size)
                rgb = (color[2], color[1], color[0])
                words = text.split()
                lines: List[str] = []
                cur: List[str] = []
                for word in words:
                    candidate = " ".join(cur + [word])
                    shaped = shape_urdu(candidate)
                    if (_measure_text_width(draw, shaped, font) <= max_width
                            or not cur):
                        cur.append(word)
                    else:
                        lines.append(" ".join(cur))
                        cur = [word]
                if cur:
                    lines.append(" ".join(cur))
                lines = lines[-max_lines:]
                for i, raw_line in enumerate(lines):
                    shaped = shape_urdu(raw_line)
                    w = _measure_text_width(draw, shaped, font)
                    draw.text((right_x - w, top_y + i * line_height),
                              shaped, font=font, fill=rgb)
        self.ops.clear()
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def filled_panel(img: np.ndarray, x0: int, y0: int, x1: int, y1: int,
                 color: Tuple[int, int, int] = COLOR_PANEL,
                 alpha: float = PANEL_ALPHA) -> np.ndarray:
    """Draw a semi-transparent filled panel in-place and return the image.

    Only the rectangle region is copied + blended (not the whole canvas),
    which keeps the per-call cost proportional to panel area instead of
    frame area. For the large right panel this is ~4x cheaper.
    """
    # Clamp coords to image bounds so slicing is safe.
    h, w = img.shape[:2]
    x0c = max(0, min(w, x0))
    y0c = max(0, min(h, y0))
    x1c = max(0, min(w, x1))
    y1c = max(0, min(h, y1))
    if x1c <= x0c or y1c <= y0c:
        return img
    region = img[y0c:y1c, x0c:x1c]
    fill = np.empty_like(region)
    fill[:] = color
    cv2.addWeighted(fill, alpha, region, 1.0 - alpha, 0.0, dst=region)
    return img


def hand_silhouette_mask(width: int, height: int) -> np.ndarray:
    """Return a grayscale mask outlining two open hands for placement hints."""
    mask = np.zeros((height, width), dtype=np.uint8)
    cx = width // 2
    cy = height // 2
    for sign in (-1, 1):
        pcx = cx + sign * int(width * 0.16)
        pcy = cy + int(height * 0.10)
        palm_w = int(width * 0.10)
        palm_h = int(height * 0.16)
        cv2.ellipse(
            mask, (pcx, pcy), (palm_w, palm_h), 0, 0, 360, 255, 4,
        )
        for dx in (-0.6, -0.2, 0.2, 0.6):
            fx = pcx + int(dx * palm_w * 1.4)
            fy_top = pcy - palm_h - int(height * 0.14)
            fy_bot = pcy - palm_h
            cv2.line(mask, (fx, fy_bot), (fx, fy_top), 255, 4)
            cv2.circle(mask, (fx, fy_top), 6, 255, 2)
        tx = pcx + sign * int(palm_w * 1.4)
        ty = pcy - int(palm_h * 0.2)
        tx2 = pcx + sign * int(palm_w * 2.2)
        ty2 = pcy - int(palm_h * 0.9)
        cv2.line(mask, (tx, ty), (tx2, ty2), 255, 4)
        cv2.circle(mask, (tx2, ty2), 6, 255, 2)
    return mask


def overlay_silhouette(img: np.ndarray, mask: np.ndarray,
                       alpha: float = SILHOUETTE_ALPHA,
                       color: Tuple[int, int, int] = COLOR_ACCENT
                       ) -> np.ndarray:
    """Blend the silhouette mask on top of `img` at the given alpha."""
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    col = np.zeros_like(img)
    col[:] = color
    sil = cv2.bitwise_and(col, col, mask=mask)
    return cv2.addWeighted(img, 1.0, sil, alpha, 0.0)


# =====================================================================
# Gesture helpers
# =====================================================================
def hand_is_open(hand_landmarks) -> bool:
    """True if 4 non-thumb fingers are extended (open palm)."""
    if hand_landmarks is None:
        return False
    lm = hand_landmarks.landmark
    checks = [
        lm[8].y < lm[6].y,
        lm[12].y < lm[10].y,
        lm[16].y < lm[14].y,
        lm[20].y < lm[18].y,
    ]
    return sum(checks) >= 4


def count_raised_fingers(hand_landmarks) -> int:
    """Count extended fingers using tip/PIP Y, plus a thumb heuristic."""
    if hand_landmarks is None:
        return 0
    lm = hand_landmarks.landmark
    n = 0
    if lm[8].y < lm[6].y:
        n += 1
    if lm[12].y < lm[10].y:
        n += 1
    if lm[16].y < lm[14].y:
        n += 1
    if lm[20].y < lm[18].y:
        n += 1
    thumb_dist = abs(lm[4].x - lm[0].x)
    mcp_dist = abs(lm[2].x - lm[0].x)
    if thumb_dist > mcp_dist + 0.02:
        n += 1
    return n


# =====================================================================
# Inference worker
# =====================================================================
@dataclass
class InferenceResult:
    """A single model output, already decoded."""
    label: str
    confidence: float
    probs: np.ndarray
    timestamp: float


class CameraReader(threading.Thread):
    """Daemon thread that continuously pulls frames from cv2.VideoCapture.

    `cap.read()` blocks until the next frame is ready, so running it on
    the main loop caps our FPS at the camera's capture rate (often 30).
    By pulling frames on a background thread and only keeping the latest
    one, the main loop never waits on I/O — at the cost of sometimes
    skipping a frame if processing is slower than capture.
    """

    def __init__(self, cap: cv2.VideoCapture,
                 stop_event: threading.Event) -> None:
        super().__init__(daemon=True, name="PSL-CameraReader")
        self._cap = cap
        self._stop_evt = stop_event
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_id: int = 0
        self._last_returned_id: int = -1

    def run(self) -> None:
        while not self._stop_evt.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame
                self._frame_id += 1

    def read(self, timeout: float = 1.0) -> Tuple[bool, Optional[np.ndarray]]:
        """Return the newest frame; blocks briefly until one arrives."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._frame is not None and self._frame_id != self._last_returned_id:
                    self._last_returned_id = self._frame_id
                    return True, self._frame
            time.sleep(0.001)
        return False, None


class InferenceWorker(threading.Thread):
    """Daemon thread that consumes normalized windows and emits predictions."""

    def __init__(self, interpreter: tf.lite.Interpreter,
                 in_det: list, out_det: list, classes: List[str],
                 in_q: "queue.Queue", out_q: "queue.Queue",
                 stop_event: threading.Event) -> None:
        """Store references; does not start the thread yet."""
        super().__init__(daemon=True, name="PSL-InferenceWorker")
        self._interp = interpreter
        self._in_det = in_det
        self._out_det = out_det
        self._classes = classes
        self._in_q = in_q
        self._out_q = out_q
        self._stop_event = stop_event
        self._buf = np.zeros((1, T_WINDOW, F_DIM), dtype=np.float32)
        self._ema: Optional[np.ndarray] = None

    def run(self) -> None:
        """Main loop: pop windows, run the model, push the result."""
        while not self._stop_event.is_set():
            try:
                window = self._in_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if window is None:
                break
            try:
                np.copyto(self._buf[0], window)
                self._interp.set_tensor(self._in_det[0]["index"], self._buf)
                self._interp.invoke()
                probs = self._interp.get_tensor(
                    self._out_det[0]["index"],
                )[0].astype(np.float32)
                if self._ema is None:
                    self._ema = probs.copy()
                else:
                    self._ema = (
                        EMA_INFER_ALPHA * self._ema
                        + (1.0 - EMA_INFER_ALPHA) * probs
                    )
                top = int(np.argmax(self._ema))
                conf = float(self._ema[top])
                label = self._classes[top]
                res = InferenceResult(
                    label=label, confidence=conf,
                    probs=self._ema.copy(), timestamp=time.time(),
                )
                try:
                    while True:
                        self._out_q.get_nowait()
                except queue.Empty:
                    pass
                self._out_q.put(res)
            except Exception as exc:
                print(f"[InferenceWorker] {exc}")

    def reset_ema(self) -> None:
        """Forget the inter-inference EMA."""
        self._ema = None


# =====================================================================
# Startup / loading screen
# =====================================================================
@dataclass
class LoadStage:
    """One labelled stage of the startup progress bar."""
    label_en: str
    label_ur: str
    done: bool = False
    failed: bool = False
    error_msg: str = ""


def draw_loading_screen(stages: List[LoadStage], active_idx: int,
                        error: Optional[str] = None) -> np.ndarray:
    """Render the full-screen startup panel for the given stages."""
    canvas = np.full(
        (DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), COLOR_BG, dtype=np.uint8,
    )
    filled_panel(canvas, 0, 0, DISPLAY_WIDTH, 140, COLOR_PANEL, 1.0)
    canvas = draw_text(canvas, (60, 35), "Listen", 62, COLOR_ACCENT)
    canvas = draw_text(
        canvas, (60, 100),
        "Pakistani Sign Language Recognition", 24, COLOR_TEXT_DIM,
    )
    canvas = draw_rtl_text(
        canvas, (DISPLAY_WIDTH - 60, 50),
        "پاکستانی اشاراتی زبان", 44, COLOR_ACCENT,
    )

    # Progress bar
    bar_x0, bar_y0 = 120, 380
    bar_x1, bar_y1 = DISPLAY_WIDTH - 120, 420
    cv2.rectangle(canvas, (bar_x0, bar_y0), (bar_x1, bar_y1),
                  COLOR_PANEL_LIGHT, -1)
    done_count = sum(1 for s in stages if s.done)
    total = max(1, len(stages))
    pct = done_count / total
    cv2.rectangle(
        canvas, (bar_x0, bar_y0),
        (bar_x0 + int((bar_x1 - bar_x0) * pct), bar_y1),
        COLOR_ACCENT, -1,
    )
    cv2.rectangle(canvas, (bar_x0, bar_y0), (bar_x1, bar_y1), COLOR_TEXT, 2)

    # Stage list
    list_x = 160
    list_y = 470
    for i, stage in enumerate(stages):
        bullet = "[x]" if stage.done else (
            "[!]" if stage.failed else (
                "[>]" if i == active_idx else "[ ]"))
        col = (COLOR_OK if stage.done else
               (COLOR_ERR if stage.failed else
                (COLOR_ACCENT if i == active_idx else COLOR_TEXT_DIM)))
        canvas = draw_text(
            canvas, (list_x, list_y + i * 48),
            f"{bullet}  {stage.label_en}", 26, col,
        )
        canvas = draw_rtl_text(
            canvas, (DISPLAY_WIDTH - list_x, list_y + i * 48),
            stage.label_ur, 26, col,
        )

    if error:
        filled_panel(
            canvas, 120, DISPLAY_HEIGHT - 180, DISPLAY_WIDTH - 120,
            DISPLAY_HEIGHT - 60, (20, 20, 60), 0.95,
        )
        canvas = draw_text(
            canvas, (140, DISPLAY_HEIGHT - 170),
            "Startup error - cannot continue", 26, COLOR_ERR,
        )
        for i, ln in enumerate(error.splitlines()[:3]):
            canvas = draw_text(
                canvas, (140, DISPLAY_HEIGHT - 130 + i * 30),
                ln, 20, COLOR_TEXT,
            )
        canvas = draw_text(
            canvas, (140, DISPLAY_HEIGHT - 80),
            "Press any key to exit.", 20, COLOR_TEXT_DIM,
        )
    return canvas


# =====================================================================
# Main application
# =====================================================================
class ListenApp:
    """The enhanced real-time PSL recognition application."""

    # ------------------------- lifecycle -----------------------------
    def __init__(self) -> None:
        """Create all state; no heavy resources loaded yet."""
        self.cap: Optional[cv2.VideoCapture] = None
        self._camera: Optional[CameraReader] = None
        self.hands = None
        self.interpreter = None
        self.in_det = None
        self.out_det = None
        self.encoder = None
        self.classes: List[str] = []

        # Thread plumbing
        self._stop_event = threading.Event()
        self._infer_in: "queue.Queue" = queue.Queue(maxsize=1)
        self._infer_out: "queue.Queue" = queue.Queue(maxsize=4)
        self._worker: Optional[InferenceWorker] = None

        # Preallocated buffers
        self._raw_buf = np.zeros((T_WINDOW, F_DIM), dtype=np.float32)
        self._norm_buf = np.zeros((T_WINDOW, F_DIM), dtype=np.float32)
        self._buf_fill = 0
        self._last_norm_frame: Optional[np.ndarray] = None
        self._prev_norm_frame: Optional[np.ndarray] = None

        # FSM
        self.state: SignState = SignState.IDLE
        self.state_since: float = time.time()
        self.cooldown_until: float = 0.0
        self.frames_since_infer: int = 0

        # Predictions / UI
        self.current_label: str = "-"
        self.current_conf_raw: float = 0.0
        self.display_conf: float = 0.0
        self.conf_threshold: float = COMMIT_CONF_DEFAULT
        self.last_result: Optional[InferenceResult] = None
        self.rolling_conf: Deque[float] = deque(maxlen=CONF_ROLL_WINDOW)
        self.low_conf_streak: int = 0
        self.no_hand_since: float = time.time()

        # Centre popup state
        self.center_popup_text: str = ""
        self.center_popup_until: float = 0.0
        self.center_popup_alpha: float = 0.0  # EMA alpha for fade

        # Session / history
        self.session_log: List[Tuple[str, str]] = []
        self.history: Deque[str] = deque(maxlen=WORD_HISTORY_MAX)

        # Undo gesture
        self._open_palm_since: Optional[float] = None
        self._undo_armed: bool = False

        # DB + suggestions
        self.db_conn = None
        self.suggestions: List[str] = []
        self._last_suggestion_query: str = ""

        # Gestural selection
        self.finger_count: int = 0
        self.finger_stable_n: int = 0
        self.FINGER_STABLE_REQUIRED: int = 15
        self.in_selection_zone: bool = False

        # Slider interaction
        self._slider_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self._dragging_slider: bool = False

        # Silhouette mask (precomputed)
        self._silhouette = hand_silhouette_mask(PROC_WIDTH, PROC_HEIGHT)

        # Motion gating
        self._last_motion: float = 0.0

        # Virtual Button States (Camera Overlay)
        self.cam_buttons = {
            "CLEAR":  {"rect": (0, 0, 0, 0), "hover": 0.0},
            "SPEAK":  {"rect": (0, 0, 0, 0), "hover": 0.0},
            "UNDO":   {"rect": (0, 0, 0, 0), "hover": 0.0},
        }
        self.cam_slider = {"rect": (0, 0, 0, 0), "active": False}
        self.HOVER_TRIGGER = 0.2
        self.suggestions_are_fallback: bool = False

        # Reusable canvas + image buffers so we don't re-allocate 3.5 MB
        # every frame. `_canvas_buf` is filled with BG at the top of each
        # compose; `_cam_big_buf` is (re)allocated only when geometry
        # changes; `_proc_rgb_buf` caches the BGR->RGB conversion target.
        self._canvas_buf: np.ndarray = np.full(
            (DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), COLOR_BG, dtype=np.uint8,
        )
        self._cam_big_buf: Optional[np.ndarray] = None
        self._proc_rgb_buf: np.ndarray = np.empty(
            (PROC_HEIGHT, PROC_WIDTH, 3), dtype=np.uint8,
        )

        # Cache the Urdu caption so we don't rebuild it every frame while
        # the sentence hasn't changed.
        self._history_cache_key: Tuple[str, ...] = ()
        self._history_urdu_cache: str = ""

        # Cache MediaPipe drawing helpers (imported once at startup
        # instead of looked up through mp.solutions.* every frame).
        self._mp_hand_connections = None
        self._mp_draw = None
        self._mp_landmarks_style = None
        self._mp_connections_style = None

    # ------------------------- startup -------------------------------
    def _init_stages(self) -> List[LoadStage]:
        """Return the ordered list of startup stages."""
        return [
            LoadStage("Initializing camera...",
                      "کیمرہ شروع کیا جا رہا ہے..."),
            LoadStage("Loading MediaPipe...",
                      "میڈیا پائپ لوڈ ہو رہا ہے..."),
            LoadStage("Loading model...", "ماڈل لوڈ ہو رہا ہے..."),
            LoadStage("Warming up inference...",
                      "انفرنس گرم کیا جا رہا ہے..."),
            LoadStage("Connecting to database...",
                      "ڈیٹا بیس سے رابطہ کیا جا رہا ہے..."),
            LoadStage("Ready", "تیار"),
        ]

    def startup(self) -> bool:
        """Run the visual startup sequence. Returns True if ready to run."""
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, DISPLAY_WIDTH, DISPLAY_HEIGHT)
        stages = self._init_stages()

        def paint(active: int, err: Optional[str] = None) -> None:
            """Render + pump the window for one frame of the loader."""
            cv2.imshow(
                WINDOW_TITLE, draw_loading_screen(stages, active, err),
            )
            cv2.waitKey(1)

        # Stage 0: camera
        paint(0)
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Camera index 0 could not be opened.")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
            # Start the async reader so the main loop never blocks on I/O.
            self._camera = CameraReader(self.cap, self._stop_event)
            self._camera.start()
            stages[0].done = True
        except Exception as e:
            stages[0].failed = True
            paint(0,
                  f"Camera not available: {e}\n"
                  f"Fix: check that a webcam is connected and not in use "
                  f"by another app.")
            cv2.waitKey(0)
            return False

        # Stage 1: MediaPipe (PROTECTED)
        paint(1)
        try:
            mp_hands_mod = mp.solutions.hands
            self.hands = mp_hands_mod.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            # Resolve drawing helpers ONCE; looking these up through
            # `mp.solutions.*` is surprisingly expensive per call.
            self._mp_hand_connections = (
                mp.solutions.hands.HAND_CONNECTIONS
            )
            self._mp_draw = mp.solutions.drawing_utils
            self._mp_landmarks_style = (
                mp.solutions.drawing_styles.get_default_hand_landmarks_style()
            )
            self._mp_connections_style = (
                mp.solutions.drawing_styles.get_default_hand_connections_style()
            )
            stages[1].done = True
        except Exception as e:
            stages[1].failed = True
            paint(1,
                  f"MediaPipe init failed: {e}\n"
                  f"Fix: `pip install mediapipe==0.10.21`.")
            cv2.waitKey(0)
            return False

        # Stage 2: model
        paint(2)
        try:
            (self.interpreter, self.in_det,
             self.out_det, self.encoder) = load_model()
            self.classes = list(self.encoder.classes_)
            stages[2].done = True
        except Exception as e:
            stages[2].failed = True
            paint(
                2,
                f"Model load failed: {e}\n"
                f"Fix: ensure models/psl_words/psl_word_classifier.tflite "
                f"and label_encoder.pkl exist.",
            )
            cv2.waitKey(0)
            return False

        # Stage 3: warm up inference thread
        paint(3)
        try:
            self._worker = InferenceWorker(
                self.interpreter, self.in_det, self.out_det,
                self.classes, self._infer_in, self._infer_out,
                self._stop_event,
            )
            self._worker.start()
            warm = np.zeros((T_WINDOW, F_DIM), dtype=np.float32)
            self._infer_in.put(warm)
            t0 = time.time()
            while time.time() - t0 < 5.0:
                try:
                    _ = self._infer_out.get(timeout=0.1)
                    break
                except queue.Empty:
                    pass
            stages[3].done = True
        except Exception as e:
            stages[3].failed = True
            paint(3, f"Inference warm-up failed: {e}")
            cv2.waitKey(0)
            return False

        # Stage 4: database (optional)
        paint(4)
        try:
            if not DB_SUPPORT:
                raise RuntimeError("psycopg2 / python-dotenv not installed")
            self.db_conn = psycopg2.connect(
                host=os.getenv("DB_HOST", "localhost"),
                port=os.getenv("DB_PORT", "5432"),
                dbname=os.getenv("DB_NAME", "urdu_dict"),
                user=os.getenv("DB_USER", "postgres"),
                password=os.getenv("DB_PASSWORD", "12345"),
            )
            stages[4].done = True
        except Exception as e:
            print(f"[WARN] DB connection failed: {e}")
            stages[4].failed = True  # non-fatal; app runs without suggestions

        stages[5].done = True
        paint(5)
        time.sleep(0.4)
        return True

    # ------------------------- mouse ---------------------------------
    def _on_mouse(self, event: int, x: int, y: int, flags: int,
                  _param) -> None:
        """Handle mouse drags on the confidence-threshold slider."""
        sx, sy, sw, sh = self._slider_rect
        inside = (sx <= x <= sx + sw) and (sy - 10 <= y <= sy + sh + 10)
        if event == cv2.EVENT_LBUTTONDOWN and inside:
            self._dragging_slider = True
        elif event == cv2.EVENT_LBUTTONUP:
            self._dragging_slider = False
        if self._dragging_slider and sw > 0:
            rel = max(0.0, min(1.0, (x - sx) / sw))
            self.conf_threshold = CONF_MIN + rel * (CONF_MAX - CONF_MIN)

    # ------------------------- FSM ----------------------------------
    def _enter_state(self, new_state: SignState) -> None:
        """Transition into `new_state` with entry side-effects."""
        if new_state is self.state:
            return
        self.state = new_state
        self.state_since = time.time()
        if new_state is SignState.COOLDOWN:
            self.cooldown_until = self.state_since + COOLDOWN_SECONDS
            self.frames_since_infer = 0
        if new_state is SignState.IDLE:
            self._buf_fill = 0
            self.frames_since_infer = 0
            if self._worker is not None:
                self._worker.reset_ema()

    # ------------------------- buffers -------------------------------
    def _push_frame(self, norm_vec: np.ndarray, raw_vec: np.ndarray) -> None:
        """Shift rolling buffers left by one and append the newest frame."""
        self._raw_buf[:-1] = self._raw_buf[1:]
        self._norm_buf[:-1] = self._norm_buf[1:]
        self._raw_buf[-1] = raw_vec
        self._norm_buf[-1] = norm_vec
        if self._buf_fill < T_WINDOW:
            self._buf_fill += 1
        self._prev_norm_frame = self._last_norm_frame
        self._last_norm_frame = norm_vec

    def _frame_motion(self) -> float:
        """Mean-squared difference of the last two normalized landmark vectors."""
        if self._last_norm_frame is None or self._prev_norm_frame is None:
            return 0.0
        d = self._last_norm_frame - self._prev_norm_frame
        return float(np.mean(d * d))

    # ------------------------- centre popup --------------------------
    def _set_center_popup(self, text: str,
                          ttl: float = PROMPT_DISMISS_SECONDS) -> None:
        """Show a light centre popup for `ttl` seconds."""
        self.center_popup_text = text
        self.center_popup_until = time.time() + ttl

    def _update_center_popup(self, has_hands: bool) -> None:
        """Show the hand-placement popup after NO_HAND_PROMPT_SECONDS.

        Dismisses immediately as soon as a hand is detected.
        """
        now = time.time()
        if has_hands:
            # Any hand on screen = the guidance popup goes away instantly.
            self.no_hand_since = now
            if self.center_popup_text == "Position your hands in the frame":
                self.center_popup_text = ""
                self.center_popup_until = 0.0
        else:
            if now - self.no_hand_since >= NO_HAND_PROMPT_SECONDS:
                # Keep refreshing the TTL while no hands; popup stays.
                self._set_center_popup(
                    "Position your hands in the frame", ttl=1.0,
                )
        # Auto-expire other popups
        if (self.center_popup_text
                and self.center_popup_text != "Position your hands in the frame"
                and now > self.center_popup_until):
            self.center_popup_text = ""

        # EMA fade towards target alpha
        target = 1.0 if self.center_popup_text else 0.0
        self.center_popup_alpha = (
            0.2 * target + 0.8 * self.center_popup_alpha
        )

    def _maybe_low_conf_popup(self, has_hands: bool) -> None:
        """Raise a low-confidence popup after a sustained poor streak."""
        if len(self.rolling_conf) < CONF_ROLL_WINDOW:
            return
        avg = float(np.mean(self.rolling_conf))
        if avg < LOW_CONF_MEAN:
            self.low_conf_streak += 1
        else:
            self.low_conf_streak = max(0, self.low_conf_streak - 1)
        if self.low_conf_streak > LOW_CONF_STREAK_THRESHOLD:
            self._set_center_popup(
                "Sign more clearly" if has_hands else "Move closer",
            )
            self.low_conf_streak = 0

    # ------------------------- undo gesture --------------------------
    def _update_undo_gesture(self, open_palm: bool) -> None:
        """Hold an open palm for UNDO_HOLD_SECONDS while IDLE to undo."""
        now = time.time()
        if self.state is not SignState.IDLE:
            self._open_palm_since = None
            self._undo_armed = False
            return
        if open_palm:
            if self._open_palm_since is None:
                self._open_palm_since = now
                self._undo_armed = False
            elif (not self._undo_armed
                  and now - self._open_palm_since >= UNDO_HOLD_SECONDS):
                self._undo_armed = True
                if self.history:
                    removed = self.history.pop()
                    self.session_log.append(
                        (datetime.now().isoformat(timespec="seconds"),
                         f"UNDO:{removed}")
                    )
        else:
            self._open_palm_since = None
            self._undo_armed = False

    # ------------------------- inference queueing -------------------
    def _maybe_enqueue(self) -> None:
        """Send the current window to the worker if conditions are met."""
        if self._buf_fill < T_WINDOW:
            return
        if self.frames_since_infer < STRIDE_FRAMES:
            return
        if self._last_motion < MOTION_VAR_MIN:
            return
        if self._infer_in.full():
            return
        snap = self._norm_buf.copy()
        self.frames_since_infer = 0
        try:
            self._infer_in.put_nowait(snap)
            self._enter_state(SignState.PREDICTING)
        except queue.Full:
            pass

    def _drain_inference(self) -> Optional[InferenceResult]:
        """Pop the freshest InferenceResult from the output queue, if any."""
        last: Optional[InferenceResult] = None
        try:
            while True:
                last = self._infer_out.get_nowait()
        except queue.Empty:
            pass
        return last

    # ------------------------- suggestions ---------------------------
    def _update_suggestions(self, full_urdu_text: str) -> None:
        """Query the DB (if available) for phrase completions or word fallbacks."""
        self.suggestions = []
        if not self.db_conn or not full_urdu_text.strip():
            return
        try:
            cur = self.db_conn.cursor()
            # 1. Try matching the entire accumulated sentence as an exact word prefix
            query = "SELECT sentence FROM urdu_sentences WHERE (sentence = %s OR sentence LIKE %s) LIMIT %s"
            cur.execute(query, (full_urdu_text, f"{full_urdu_text} %", SUGGESTIONS_MAX))
            rows = cur.fetchall()
            
            if rows:
                self.suggestions_are_fallback = False
            else:
                # 2. If no full match, fallback to matching just the LAST word exactly
                self.suggestions_are_fallback = True
                words = full_urdu_text.split()
                if words:
                    last_word = words[-1]
                    cur.execute(query, (last_word, f"{last_word} %", SUGGESTIONS_MAX))
                    rows = cur.fetchall()
            
            self.suggestions = [r[0] for r in rows]
            cur.close()
        except Exception as exc:
            print(f"[DB] {exc}")
            try:
                self.db_conn.rollback()
            except Exception:
                pass

    def _commit_suggestion(self, selected_sentence: str) -> None:
        """Accept a DB suggestion. If fallback, replace only the last word."""
        if self.suggestions_are_fallback and self.history:
            self.history.pop()
        else:
            self.history.clear()
        
        self.history.append(selected_sentence)
        self.session_log.append(
            (datetime.now().isoformat(timespec="seconds"),
             f"SUGGESTION:{selected_sentence}")
        )
        
        # Refresh for the new state
        cur_sent = " ".join(PSL_WORD_TO_URDU.get(w, w) for w in self.history)
        self._last_suggestion_query = cur_sent
        self._update_suggestions(cur_sent)

        speak_text_async(selected_sentence)
        self._enter_state(SignState.COOLDOWN)

    # ------------------------- cached sentence -----------------------
    def _get_urdu_sentence(self) -> str:
        """Return the running Urdu caption; rebuilds only on history change."""
        key = tuple(self.history)
        if key != self._history_cache_key:
            self._history_cache_key = key
            self._history_urdu_cache = " ".join(
                PSL_WORD_TO_URDU.get(w, w) for w in self.history
            )
        return self._history_urdu_cache

    # ------------------------- commit --------------------------------
    def _commit_word(self, english_word: str) -> None:
        """Append a word, fire TTS, and lock into COMMITTED state."""
        if english_word in IDLE_CLASSES:
            return
        self._enter_state(SignState.COMMITTED)
        self.history.append(english_word)
        self.session_log.append(
            (datetime.now().isoformat(timespec="seconds"), english_word),
        )
        speak_word_async(english_word)

    # ------------------------- main loop -----------------------------
    def run(self) -> None:
        """Main capture + render loop."""
        cv2.setMouseCallback(WINDOW_TITLE, self._on_mouse)
        assert self.cap is not None and self.hands is not None
        assert self._camera is not None

        proc_frame_bgr = np.zeros(
            (PROC_HEIGHT, PROC_WIDTH, 3), dtype=np.uint8,
        )
        fps_t0 = time.time()
        fps_n = 0
        fps = 0.0
        # Per-stage timing accumulators (ms). Reset every 60 frames.
        t_cam = t_mp = t_logic = t_compose = t_show = 0.0
        probe_n = 0
        perf = time.perf_counter
        print(f"[{datetime.now().strftime('%H:%M:%S')}] App started.")

        while not self._stop_event.is_set():
            # Trigger DB suggestion refresh only if history actually changed
            # (avoids rebuilding the Urdu sentence string every frame).
            urdu_sentence = self._get_urdu_sentence()
            if urdu_sentence != self._last_suggestion_query:
                self._update_suggestions(urdu_sentence)
                self._last_suggestion_query = urdu_sentence

            _ts = perf()
            ok, frame_bgr = self._camera.read()
            if not ok or frame_bgr is None:
                continue
            t_cam += (perf() - _ts) * 1000.0

            _ts = perf()
            # Downscale to processing resolution
            cv2.resize(
                frame_bgr, (PROC_WIDTH, PROC_HEIGHT), dst=proc_frame_bgr,
                interpolation=cv2.INTER_LINEAR,
            )
            # Reuse the preallocated RGB buffer instead of allocating one
            # every frame.
            cv2.cvtColor(proc_frame_bgr, cv2.COLOR_BGR2RGB,
                         dst=self._proc_rgb_buf)
            results = _HandsToHolistic(self.hands.process(self._proc_rgb_buf))
            t_mp += (perf() - _ts) * 1000.0
            _ts = perf()

            has_left = results.left_hand_landmarks is not None
            has_right = results.right_hand_landmarks is not None
            has_hands = has_left or has_right
            open_palm = (
                hand_is_open(results.left_hand_landmarks)
                or hand_is_open(results.right_hand_landmarks)
            )

            # Gestural selection (left edge of the processing frame)
            self.finger_count = 0
            self.in_selection_zone = False
            wrist_x = -1
            hlm_sel = None
            if results.left_hand_landmarks:
                wrist_x = results.left_hand_landmarks.landmark[0].x * PROC_WIDTH
                hlm_sel = results.left_hand_landmarks
            elif results.right_hand_landmarks:
                wrist_x = results.right_hand_landmarks.landmark[0].x * PROC_WIDTH
                hlm_sel = results.right_hand_landmarks
            if wrist_x != -1 and wrist_x < PROC_WIDTH * 0.25:
                self.in_selection_zone = True
                self.finger_count = count_raised_fingers(hlm_sel)
                if self.finger_count > 0:
                    self.finger_stable_n += 1
                else:
                    self.finger_stable_n = 0
            else:
                self.finger_stable_n = 0

            # Feature extraction
            raw_vec = frame_from_mediapipe(results)
            norm_vec = normalize_frame(raw_vec)
            self._push_frame(norm_vec, raw_vec)
            self._last_motion = self._frame_motion()
            self.frames_since_infer += 1

            # FSM transitions
            now = time.time()
            if self.state is SignState.IDLE:
                if has_hands and self._last_motion > MOTION_VAR_MIN:
                    self._enter_state(SignState.SIGNING)
            elif self.state is SignState.SIGNING:
                if (not has_hands
                        and self._last_motion < MOTION_VAR_MIN
                        and now - self.state_since > 1.2):
                    self._enter_state(SignState.IDLE)
                else:
                    self._maybe_enqueue()
            elif self.state is SignState.PREDICTING:
                res = self._drain_inference()
                if res is not None:
                    self.last_result = res
                    self.current_label = res.label
                    self.current_conf_raw = res.confidence
                    self.rolling_conf.append(res.confidence)
                    if (res.label not in IDLE_CLASSES
                            and res.confidence >= self.conf_threshold):
                        self._commit_word(res.label)
                    else:
                        self._enter_state(SignState.SIGNING)
            elif self.state is SignState.COMMITTED:
                self._enter_state(SignState.COOLDOWN)
            elif self.state is SignState.COOLDOWN:
                if has_hands:
                    self.cooldown_until = time.time() + COOLDOWN_SECONDS
                if time.time() >= self.cooldown_until and not has_hands:
                    self._enter_state(SignState.IDLE)

            # Gestural suggestion selection (1-5 fingers = pick that suggestion)
            if self.finger_stable_n >= self.FINGER_STABLE_REQUIRED:
                idx = self.finger_count - 1
                if 0 <= idx < len(self.suggestions):
                    self._commit_suggestion(self.suggestions[idx])
                self.finger_stable_n = -15  # cooldown counter

            # Display confidence smoothing
            self.display_conf = (
                EMA_DISPLAY_ALPHA * self.current_conf_raw
                + (1.0 - EMA_DISPLAY_ALPHA) * self.display_conf
            )

            # Updates
            self._update_undo_gesture(open_palm)
            self._update_center_popup(has_hands)
            self._maybe_low_conf_popup(has_hands)
            self._update_cam_interact(results)

            # Draw landmarks onto proc frame
            self._draw_landmarks(proc_frame_bgr, results)

            # FPS (computed before compose so it can be batched with
            # the rest of the frame's text into one PIL round-trip)
            fps_n += 1
            if fps_n >= 10:
                dt = time.time() - fps_t0
                fps = fps_n / dt if dt > 0 else 0.0
                fps_t0 = time.time()
                fps_n = 0

            t_logic += (perf() - _ts) * 1000.0

            # Compose final canvas
            _ts = perf()
            canvas = self._compose(proc_frame_bgr, has_hands, fps=fps)
            t_compose += (perf() - _ts) * 1000.0

            _ts = perf()
            cv2.imshow(WINDOW_TITLE, canvas)
            key = cv2.waitKey(1) & 0xFF
            t_show += (perf() - _ts) * 1000.0

            probe_n += 1
            if probe_n >= 60:
                print(
                    f"[perf] cam={t_cam/probe_n:.1f}ms "
                    f"mp={t_mp/probe_n:.1f}ms "
                    f"logic={t_logic/probe_n:.1f}ms "
                    f"compose={t_compose/probe_n:.1f}ms "
                    f"show={t_show/probe_n:.1f}ms "
                    f"fps={fps:.1f}",
                    flush=True,
                )
                t_cam = t_mp = t_logic = t_compose = t_show = 0.0
                probe_n = 0

            if key in (ord('q'), 27):
                self._stop_event.set()
                break

        self._shutdown()

    # ------------------------- landmarks -----------------------------
    def _draw_landmarks(self, proc_bgr: np.ndarray, results) -> None:
        """Overlay hand landmarks in place on the processing-res frame."""
        for hlm in (results.left_hand_landmarks, results.right_hand_landmarks):
            if hlm is not None:
                draw_hand_fast(proc_bgr, hlm)

    def _draw_cam_btn(self, canvas: np.ndarray, name: str,
                      color: Tuple[int, int, int]) -> None:
        """Draw a virtual button on the camera section."""
        x, y, w, h = self.cam_buttons[name]["rect"]
        alpha = self.cam_buttons[name]["hover"] / self.HOVER_TRIGGER
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (30, 30, 40), -1)
        if alpha > 0:
            cv2.rectangle(canvas, (x, y), (x + int(w * alpha), y + h),
                          color, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (200, 200, 200), 1)
        cv2.putText(canvas, name, (x + 15, y + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _update_cam_interact(self, results) -> None:
        """Detect finger-tip hover over virtual buttons or the slider area."""
        tip = None
        hand = results.right_hand_landmarks or results.left_hand_landmarks
        if hand:
            # Mirror logic: user right (0..0.25) -> canvas-X near cam_x1
            raw_x = hand.landmark[8].x
            left_w = int(DISPLAY_WIDTH * LEFT_PANEL_FRAC)
            tip_x = (1.0 - raw_x) * (left_w - 40) + 20
            tip_y = hand.landmark[8].y * (DISPLAY_HEIGHT - 170 - 40) + 40
            tip = (int(tip_x), int(tip_y))

        dt = 0.033
        for name, btn in self.cam_buttons.items():
            bx, by, bw, bh = btn["rect"]
            if tip and bx <= tip[0] <= bx + bw and by <= tip[1] <= by + bh:
                btn["hover"] += dt
                if btn["hover"] >= self.HOVER_TRIGGER:
                    self._trigger_cam_btn(name)
                    btn["hover"] = -1.0
            else:
                btn["hover"] = max(0.0, btn["hover"] - dt * 2)

    def _trigger_cam_btn(self, name: str) -> None:
        """Execute the action corresponding to a virtual button."""
        if name == "CLEAR":
            self.history.clear()
        elif name == "SPEAK":
            if self.history:
                speak_text_async(self._get_urdu_sentence())
        elif name == "UNDO":
            if self.history:
                self.history.pop()

    # ------------------------- compose UI ----------------------------
    def _compose(self, proc_bgr: np.ndarray, has_hands: bool,
                 fps: float = 0.0) -> np.ndarray:
        """Build and return the full two-panel UI frame for this tick."""
        # Reuse the same canvas buffer every frame; fill() is faster than
        # allocating a fresh (1440*810*3)-byte array each tick.
        canvas = self._canvas_buf
        canvas[:] = COLOR_BG
        batch = _TextBatch()

        # Panel geometry
        left_w = int(DISPLAY_WIDTH * LEFT_PANEL_FRAC)
        right_w = DISPLAY_WIDTH - left_w

        # Reserve bottom strip of the left panel for the flowing caption.
        caption_h = CAPTION_LINE_HEIGHT * CAPTION_MAX_LINES + 30
        cam_x0 = 20
        cam_y0 = 40
        cam_x1 = left_w - 20
        cam_y1 = DISPLAY_HEIGHT - caption_h - 40
        cam_area_w = cam_x1 - cam_x0
        cam_area_h = cam_y1 - cam_y0

        # Silhouette only when IDLE + no hands
        proc_draw = proc_bgr
        if self.state is SignState.IDLE and not has_hands:
            proc_draw = overlay_silhouette(proc_bgr, self._silhouette)

        # Reuse a single cam_big buffer; re-alloc only if geometry changes
        # (which happens at most once after startup).
        if (self._cam_big_buf is None
                or self._cam_big_buf.shape[0] != cam_area_h
                or self._cam_big_buf.shape[1] != cam_area_w):
            self._cam_big_buf = np.empty(
                (cam_area_h, cam_area_w, 3), dtype=np.uint8,
            )
        cv2.resize(proc_draw, (cam_area_w, cam_area_h),
                   dst=self._cam_big_buf,
                   interpolation=cv2.INTER_LINEAR)
        cv2.flip(self._cam_big_buf, 1, dst=self._cam_big_buf)
        canvas[cam_y0:cam_y1, cam_x0:cam_x1] = self._cam_big_buf

        # Camera border + label
        cv2.rectangle(canvas, (cam_x0 - 4, cam_y0 - 4),
                      (cam_x1 + 4, cam_y1 + 4), COLOR_PANEL_LIGHT, 2)
        batch.draw((cam_x0, 10), "Live camera", 20, COLOR_TEXT_DIM)

        # ---- Centre popup (light, auto-dismiss when hands appear) ----
        if self.center_popup_alpha > 0.02 and self.center_popup_text:
            popup_text = self.center_popup_text
            # Size the popup to its text using the offscreen measurer
            # so we don't convert the full canvas just to measure.
            font = get_font(26)
            tw = _measure_text_width(_MEASURE_DRAW, popup_text, font)
            pad_x, pad_y = 28, 14
            pw = tw + 2 * pad_x
            ph = 56
            px0 = (cam_x0 + cam_x1) // 2 - pw // 2
            py0 = (cam_y0 + cam_y1) // 2 - ph // 2
            px1 = px0 + pw
            py1 = py0 + ph
            a = 0.55 * self.center_popup_alpha
            overlay = canvas.copy()
            cv2.rectangle(overlay, (px0, py0), (px1, py1), (25, 25, 35), -1)
            cv2.addWeighted(overlay, a, canvas, 1 - a, 0.0, dst=canvas)
            cv2.rectangle(canvas, (px0, py0), (px1, py1),
                          COLOR_PANEL_LIGHT, 1)
            batch.draw((px0 + pad_x, py0 + pad_y),
                       popup_text, 26, COLOR_TEXT)

        # ---- Undo progress ring (small, bottom-left of camera) ------
        if (self.state is SignState.IDLE
                and self._open_palm_since is not None):
            held = min(UNDO_HOLD_SECONDS,
                       time.time() - self._open_palm_since)
            frac = held / UNDO_HOLD_SECONDS
            cx = cam_x0 + 60
            cy = cam_y1 - 60
            cv2.circle(canvas, (cx, cy), 34, COLOR_PANEL_LIGHT, 2)
            cv2.ellipse(canvas, (cx, cy), (34, 34), -90,
                        0, int(360 * frac), COLOR_WARN, 4)
            batch.draw((cx - 24, cy - 12), "UNDO", 14, COLOR_TEXT)

        # ---- Virtual Buttons (Top of Camera) ----
        btn_y = cam_y0 + 20
        btn_w = 110
        btn_h = 50
        
        # CLEAR
        self.cam_buttons["CLEAR"]["rect"] = (cam_x0 + 20, btn_y, btn_w, btn_h)
        self._draw_cam_btn(canvas, "CLEAR", COLOR_ERR)
        
        # SPEAK
        self.cam_buttons["SPEAK"]["rect"] = (cam_x0 + 150, btn_y, btn_w, btn_h)
        self._draw_cam_btn(canvas, "SPEAK", COLOR_OK)

        # UNDO (Visual Shortcut)
        self.cam_buttons["UNDO"]["rect"] = (cam_x0 + 280, btn_y, btn_w, btn_h)
        self._draw_cam_btn(canvas, "UNDO", COLOR_WARN)

        # ---- Caption strip under the camera (flowing, wrapped) ------
        cap_y0 = cam_y1 + 16
        cap_y1 = DISPLAY_HEIGHT - 24
        filled_panel(canvas, cam_x0, cap_y0, cam_x1, cap_y1,
                     COLOR_PANEL, 0.55)
        # The caption is the running sentence in Urdu, wrapped RTL.
        urdu_sentence = self._get_urdu_sentence()
        if urdu_sentence:
            batch.wrap(
                right_x=cam_x1 - 18,
                top_y=cap_y0 + 12,
                text=urdu_sentence,
                size=CAPTION_FONT_SIZE,
                max_width=cam_area_w - 36,
                line_height=CAPTION_LINE_HEIGHT,
                max_lines=CAPTION_MAX_LINES,
                color=COLOR_TEXT,
            )
        else:
            batch.draw(
                (cam_x0 + 18, cap_y0 + 18),
                "Your sentence will appear here.",
                20, COLOR_TEXT_DIM,
            )

        # ====================== RIGHT PANEL ==========================
        rp_x0 = left_w
        rp_y0 = 0
        rp_x1 = DISPLAY_WIDTH
        rp_y1 = DISPLAY_HEIGHT
        filled_panel(canvas, rp_x0, rp_y0, rp_x1, rp_y1,
                     COLOR_PANEL, PANEL_ALPHA)

        # --- Current word (English + Urdu) ---
        batch.draw((rp_x0 + 24, 24), "Current word", 18, COLOR_TEXT_DIM)
        label_disp = (self.current_label
                      if self.current_label not in IDLE_CLASSES
                      else "-")
        batch.draw((rp_x0 + 24, 48), label_disp, 32, COLOR_ACCENT)
        urdu = PSL_WORD_TO_URDU.get(label_disp, "-")
        batch.rtl((rp_x1 - 24, 48), urdu, 52, COLOR_TEXT)

        # --- Confidence bar ---
        bar_x = rp_x0 + 24
        bar_y = 150
        bar_w = right_w - 48
        bar_h = 22
        cv2.rectangle(canvas, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h),
                      COLOR_PANEL_LIGHT, -1)
        conf = max(0.0, min(1.0, self.display_conf))
        conf_col = COLOR_OK if conf >= self.conf_threshold else COLOR_WARN
        cv2.rectangle(canvas, (bar_x, bar_y),
                      (bar_x + int(bar_w * conf), bar_y + bar_h),
                      conf_col, -1)
        cv2.rectangle(canvas, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h),
                      COLOR_TEXT_DIM, 1)
        tx = bar_x + int(bar_w * self.conf_threshold)
        cv2.line(canvas, (tx, bar_y - 6),
                 (tx, bar_y + bar_h + 6), COLOR_ERR, 2)
        batch.draw((bar_x, bar_y - 22),
                   f"Confidence  {int(conf * 100)}%", 18, COLOR_TEXT)

        # --- State indicator ---
        state_lbl = STATE_LABEL[self.state]
        state_col = STATE_COLORS[state_lbl]
        sy = 210
        cv2.circle(canvas, (rp_x0 + 36, sy + 12), 10, state_col, -1)
        batch.draw((rp_x0 + 56, sy), f"State: {state_lbl}", 22, COLOR_TEXT)
        if self.state is SignState.COOLDOWN:
            remaining = max(0.0, self.cooldown_until - time.time())
            batch.draw((rp_x0 + 56, sy + 28),
                       f"Ready in {remaining:0.1f}s", 18, COLOR_TEXT_DIM)
        elif self.state is SignState.PREDICTING:
            batch.draw((rp_x0 + 56, sy + 28),
                       "Analyzing...", 18, COLOR_TEXT_DIM)
        elif self.state is SignState.SIGNING:
            frac = min(1.0, self._buf_fill / T_WINDOW)
            batch.draw((rp_x0 + 56, sy + 28),
                       f"Capturing  {int(frac * 100)}%", 18, COLOR_TEXT_DIM)
        elif self.state is SignState.COMMITTED:
            batch.draw((rp_x0 + 56, sy + 28),
                       "Word committed", 18, COLOR_OK)
        else:
            batch.draw((rp_x0 + 56, sy + 28),
                       "Waiting for hands", 18, COLOR_TEXT_DIM)

        # --- Confidence threshold slider (mid-panel) ---
        slider_x = rp_x0 + 24
        slider_y = 300
        slider_w = right_w - 48
        slider_h = 8
        batch.draw(
            (slider_x, slider_y - 26),
            f"Confidence threshold  "
            f"{int(self.conf_threshold * 100)}%",
            18, COLOR_TEXT,
        )
        cv2.rectangle(canvas, (slider_x, slider_y),
                      (slider_x + slider_w, slider_y + slider_h),
                      COLOR_PANEL_LIGHT, -1)
        rel = ((self.conf_threshold - CONF_MIN)
               / max(1e-9, CONF_MAX - CONF_MIN))
        knob_x = slider_x + int(slider_w * rel)
        cv2.circle(canvas, (knob_x, slider_y + slider_h // 2),
                   11, COLOR_ACCENT, -1)
        cv2.circle(canvas, (knob_x, slider_y + slider_h // 2),
                   11, COLOR_TEXT, 1)
        self._slider_rect = (slider_x, slider_y, slider_w, slider_h)

        # --- Suggestions (replaces history on the right panel) ------
        sug_y = 360
        batch.draw((rp_x0 + 24, sug_y), "Suggestions", 20, COLOR_TEXT_DIM)
        row_h = 56
        if not self.suggestions:
            batch.draw(
                (rp_x0 + 24, sug_y + 34),
                ("Keep signing to see phrase suggestions."
                 if self.db_conn
                 else "Suggestions unavailable (no dictionary DB)."),
                16, COLOR_TEXT_DIM,
            )
        else:
            for i, sug in enumerate(self.suggestions[:SUGGESTIONS_MAX]):
                row_y = sug_y + 30 + i * row_h
                # Row background with a subtle highlight if user is aiming
                # this row via finger count.
                hi = (self.in_selection_zone
                      and self.finger_count == i + 1)
                row_bg = (60, 60, 85) if hi else COLOR_PANEL_LIGHT
                cv2.rectangle(
                    canvas,
                    (rp_x0 + 20, row_y - 6),
                    (rp_x1 - 20, row_y + row_h - 10),
                    row_bg, -1,
                )
                cv2.rectangle(
                    canvas,
                    (rp_x0 + 20, row_y - 6),
                    (rp_x1 - 20, row_y + row_h - 10),
                    (90, 90, 110), 1,
                )
                batch.draw((rp_x0 + 30, row_y - 2),
                           f"{i + 1}.", 18, COLOR_ACCENT)
                batch.rtl((rp_x1 - 30, row_y - 4), sug, 24, COLOR_TEXT)
                if hi and self.finger_stable_n >= 0:
                    frac = min(
                        1.0,
                        self.finger_stable_n / self.FINGER_STABLE_REQUIRED,
                    )
                    cv2.rectangle(
                        canvas,
                        (rp_x0 + 20, row_y + row_h - 12),
                        (rp_x0 + 20 + int((right_w - 40) * frac),
                         row_y + row_h - 10),
                        COLOR_ACCENT, -1,
                    )

        # FPS indicator (top-right), folded into the same batch so the
        # whole frame's text costs one PIL conversion total.
        if fps > 0.0:
            batch.draw(
                (DISPLAY_WIDTH - 180, 14),
                f"FPS {fps:4.1f}", 18, COLOR_TEXT_DIM,
            )

        # Single PIL round-trip for all text drawn this frame.
        canvas = batch.flush(canvas)
        return canvas

    def _shutdown(self) -> None:
        """Release resources, stop threads, and cleanup."""
        self._stop_event.set()
        if self._camera is not None:
            self._camera.join(timeout=1.0)
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        try:
            if self.hands is not None:
                self.hands.close()
        except Exception:
            pass
        try:
            self._infer_in.put_nowait(None)
        except Exception:
            pass
        if self._worker is not None:
            self._worker.join(timeout=1.0)
        try:
            if self.db_conn is not None:
                self.db_conn.close()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


# =====================================================================
# Entry point
# =====================================================================
def main() -> int:
    """Run the Listen PSL application; return a shell-style exit code."""
    app = ListenApp()
    try:
        if not app.startup():
            return 1
        app.run()
        return 0
    except KeyboardInterrupt:
        app._shutdown()
        return 0
    except Exception as exc:
        print(f"[FATAL] {exc}")
        app._shutdown()
        return 2


if __name__ == "__main__":
    sys.exit(main())
