"""Per-connection PSL recognition session.

Mirrors the inference pipeline in src/inference/psl_words_v2.py exactly so
the mobile app gets desktop-parity accuracy. Constants, MediaPipe init,
feature extraction, normalization, FSM and EMA smoothing are all 1:1.
"""

from __future__ import annotations

import os
import time
from collections import deque
from enum import Enum, auto
from typing import Optional

import cv2
import joblib
import mediapipe as mp
import numpy as np
import tensorflow as tf

# ── Constants (match psl_words_v2.py) ──────────────────────────────────
T_WINDOW: int = 60
F_DIM: int = 126
HAND_COUNT: int = 2
LANDMARKS_PER_HAND: int = 21
COORDS_PER_LANDMARK: int = 3

STRIDE_FRAMES: int = 3
COMMIT_CONF_DEFAULT: float = 0.70
COOLDOWN_SECONDS: float = 0.8
MOTION_VAR_MIN: float = 1e-4
EMA_INFER_ALPHA: float = 0.60
SIGNING_QUIET_SECS: float = 1.2

# Selfie-trained MediaPipe → label "Right" is anatomical left.
HANDS_INVERT_HANDEDNESS: bool = True

IDLE_CLASSES = {"nothing", "test_word"}

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


# ── Paths ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
MODEL_PATH = os.path.join(_REPO_ROOT, "models", "psl_words", "psl_word_classifier.tflite")
ENCODER_PATH = os.path.join(_REPO_ROOT, "models", "psl_words", "label_encoder.pkl")


# ── Module-level model singletons ──────────────────────────────────────
# TFLite interpreter is not safe for concurrent invoke() across threads
# without a lock. We keep one shared interpreter and serialize calls in
# SignSession with a class-level lock.
import threading

_model_lock = threading.Lock()
_interpreter: Optional[tf.lite.Interpreter] = None
_in_idx: int = 0
_out_idx: int = 0
_label_encoder = None
_class_labels: list[str] = []


def _ensure_model_loaded() -> None:
    global _interpreter, _in_idx, _out_idx, _label_encoder, _class_labels
    if _interpreter is not None:
        return
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Encoder not found: {ENCODER_PATH}")
    interp = tf.lite.Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()
    in_det = interp.get_input_details()
    interp.resize_tensor_input(in_det[0]["index"], [1, T_WINDOW, F_DIM])
    interp.allocate_tensors()
    in_det = interp.get_input_details()
    out_det = interp.get_output_details()
    _interpreter = interp
    _in_idx = in_det[0]["index"]
    _out_idx = out_det[0]["index"]
    _label_encoder = joblib.load(ENCODER_PATH)
    _class_labels = list(_label_encoder.classes_)
    print(f"[server] model loaded: {len(_class_labels)} classes")


# ── FSM ────────────────────────────────────────────────────────────────
class SignState(Enum):
    IDLE = auto()
    SIGNING = auto()
    PREDICTING = auto()
    COMMITTED = auto()
    COOLDOWN = auto()


_STATE_LABEL = {
    SignState.IDLE: "IDLE",
    SignState.SIGNING: "SIGNING",
    SignState.PREDICTING: "PREDICTING",
    SignState.COMMITTED: "COMMITTED",
    SignState.COOLDOWN: "COOLDOWN",
}


# ── Landmark helpers (1:1 with psl_words_v2.py) ────────────────────────
class _HandsToHolistic:
    __slots__ = ("left_hand_landmarks", "right_hand_landmarks")

    def __init__(self, results) -> None:
        self.left_hand_landmarks = None
        self.right_hand_landmarks = None
        mhl = getattr(results, "multi_hand_landmarks", None)
        if not mhl:
            return
        mhd = getattr(results, "multi_handedness", None) or []
        for lm, hd in zip(mhl, mhd):
            label = hd.classification[0].label
            is_anatomical_left = (
                label == "Right" if HANDS_INVERT_HANDEDNESS else label == "Left"
            )
            if is_anatomical_left:
                if self.left_hand_landmarks is None:
                    self.left_hand_landmarks = lm
            else:
                if self.right_hand_landmarks is None:
                    self.right_hand_landmarks = lm


def _frame_from_mediapipe(results) -> np.ndarray:
    def flat(lm_list) -> np.ndarray:
        if lm_list is None:
            return np.zeros(LANDMARKS_PER_HAND * COORDS_PER_LANDMARK, dtype=np.float32)
        return np.array(
            [[p.x, p.y, p.z] for p in lm_list.landmark],
            dtype=np.float32,
        ).reshape(-1)
    return np.concatenate([flat(results.left_hand_landmarks),
                           flat(results.right_hand_landmarks)])


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    f = frame.reshape(HAND_COUNT, LANDMARKS_PER_HAND, COORDS_PER_LANDMARK).copy()
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


# ── SignSession ────────────────────────────────────────────────────────
class SignSession:
    """One per WebSocket connection. Not thread-safe — call from one task."""

    def __init__(self, conf_threshold: float = COMMIT_CONF_DEFAULT) -> None:
        _ensure_model_loaded()
        self.conf_threshold = conf_threshold

        # MediaPipe Hands — model_complexity=0 uses hand_landmark_lite,
        # which is what the classifier was trained on. This is the exact
        # init from psl_words_v2.py:1091-1097.
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Rolling buffer (deque of normalized 126-D frames).
        self._buf: deque[np.ndarray] = deque(maxlen=T_WINDOW)
        self._prev_norm: Optional[np.ndarray] = None
        self._last_motion: float = 0.0

        # FSM
        self._state = SignState.IDLE
        self._state_since = time.time()
        self._frames_since_infer = 0
        self._cooldown_until = 0.0

        # EMA smoothing of class probabilities.
        self._ema_probs: Optional[np.ndarray] = None
        self._last_label = ""
        self._last_conf = 0.0
        self._committed_this_frame = False

    def close(self) -> None:
        try:
            self._hands.close()
        except Exception:
            pass

    # ── Public API ─────────────────────────────────────────────────────
    def process_jpeg(self, jpeg_bytes: bytes) -> dict:
        """Decode a JPEG frame and step the pipeline.

        Returns a dict ready to be JSON-serialized to the client.
        """
        # Decode JPEG → BGR ndarray.
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return self._snapshot(has_hands=False, error="bad_jpeg")

        # MediaPipe expects RGB.
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        raw = self._hands.process(rgb)
        results = _HandsToHolistic(raw)
        has_hands = (
            results.left_hand_landmarks is not None
            or results.right_hand_landmarks is not None
        )

        # Always extract + push, even when no hands (zero vector). Skipping
        # makes the 60-frame window span a longer wall-clock time than the
        # model was trained on. Matches psl_words_v2.py:1493.
        raw_vec = _frame_from_mediapipe(results)
        norm_vec = _normalize_frame(raw_vec)
        self._buf.append(norm_vec)
        self._last_motion = self._motion(norm_vec)
        self._prev_norm = norm_vec
        self._frames_since_infer += 1

        # FSM. Server inference is synchronous, so PREDICTING/COMMITTED
        # are collapsed: SIGNING runs inference inline; on commit we jump
        # straight to COOLDOWN and flag this frame as the commit event.
        now = time.time()
        self._committed_this_frame = False
        if self._state is SignState.IDLE:
            if has_hands and self._last_motion > MOTION_VAR_MIN:
                self._enter(SignState.SIGNING, now)
        elif self._state is SignState.SIGNING:
            if (not has_hands
                    and self._last_motion < MOTION_VAR_MIN
                    and now - self._state_since > SIGNING_QUIET_SECS):
                self._enter(SignState.IDLE, now)
            else:
                self._maybe_infer(now)
        elif self._state is SignState.COOLDOWN:
            if has_hands:
                self._cooldown_until = now + COOLDOWN_SECONDS
            if now >= self._cooldown_until and not has_hands:
                self._enter(SignState.IDLE, now)
                self._buf.clear()
                self._ema_probs = None

        return self._snapshot(has_hands=has_hands)

    # ── Internals ──────────────────────────────────────────────────────
    def _motion(self, norm_vec: np.ndarray) -> float:
        if self._prev_norm is None:
            return 0.0
        d = norm_vec - self._prev_norm
        return float(np.mean(d * d))

    def _enter(self, s: SignState, now: float) -> None:
        if self._state is s:
            return
        self._state = s
        self._state_since = now

    def _maybe_infer(self, now: float) -> None:
        if len(self._buf) < T_WINDOW:
            return
        if self._frames_since_infer < STRIDE_FRAMES:
            return
        if self._last_motion < MOTION_VAR_MIN:
            return
        self._frames_since_infer = 0

        window = np.stack(self._buf, axis=0).astype(np.float32)  # (60, 126)
        window = window.reshape(1, T_WINDOW, F_DIM)

        with _model_lock:
            _interpreter.set_tensor(_in_idx, window)
            _interpreter.invoke()
            probs = _interpreter.get_tensor(_out_idx)[0].astype(np.float32)

        if self._ema_probs is None:
            self._ema_probs = probs.copy()
        else:
            self._ema_probs = (
                EMA_INFER_ALPHA * probs
                + (1.0 - EMA_INFER_ALPHA) * self._ema_probs
            )

        top = int(np.argmax(self._ema_probs))
        conf = float(self._ema_probs[top])
        label = _class_labels[top] if 0 <= top < len(_class_labels) else ""

        self._last_label = label
        self._last_conf = conf

        if (label not in IDLE_CLASSES) and conf >= self.conf_threshold:
            self._committed_this_frame = True
            self._cooldown_until = now + COOLDOWN_SECONDS
            self._enter(SignState.COOLDOWN, now)
        # Else: stay in SIGNING; no state change needed.

    def _snapshot(self, has_hands: bool, error: Optional[str] = None) -> dict:
        urdu = PSL_WORD_TO_URDU.get(self._last_label, self._last_label)
        return {
            "state": _STATE_LABEL[self._state],
            "label": self._last_label,
            "english": self._last_label,
            "urdu": urdu,
            "confidence": float(self._last_conf),
            "committed": bool(self._committed_this_frame),
            "hasHands": bool(has_hands),
            "bufferFill": len(self._buf),
            "bufferCapacity": T_WINDOW,
            "error": error,
        }
