"""Per-connection PSL alphabet recognition session.

Mirrors src/inference/psl-v1.py: single-frame, 42-D landmark vector
(x,y only), StandardScaler-normalized, MediaPipe Hands with max_num_hands=1.
A letter is committed after STABLE_REQUIRED consecutive frames of the same
prediction at conf >= THRESHOLD.
"""

from __future__ import annotations

import os
import threading
import time
from enum import Enum, auto
from typing import Optional

import cv2
import joblib
import mediapipe as mp
import numpy as np
import tensorflow as tf

# ── Constants (match psl-v1.py) ────────────────────────────────────────
THRESHOLD: float = 0.85
STABLE_REQUIRED: int = 40
COOLDOWN_SECONDS: float = 0.4
LANDMARKS_PER_HAND: int = 21
F_DIM: int = 42  # 21 × (x, y)

PSL_TO_URDU = {
    "Ain": "ع", "Alif": "ا", "Alifmad": "آ", "Aray": "ڑ",
    "Bay": "ب", "Byeh": "ے", "Chay": "چ", "Cyeh": "ی",
    "Daal": "ڈ", "Dal": "د", "Dochahay": "ھ", "Fay": "ف",
    "Gaaf": "گ", "Ghain": "غ", "Hamza": "ء", "Hay": "ح",
    "Jeem": "ج", "Kaf": "ک", "Khay": "خ", "Kiaf": "ق",
    "Lam": "ل", "Meem": "م", "Nuun": "ن", "Nuungh": "ں",
    "Pay": "پ", "Ray": "ر", "Say": "ث", "Seen": "س",
    "Sheen": "ش", "Suad": "ص", "Taay": "ط", "Tay": "ت",
    "Tuey": "ٹ", "Wao": "و", "Zaal": "ذ", "Zaey": "ی",
    "Zay": "ز", "Zuad": "ض", "Zuey": "ظ",
}


# ── Paths ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
ALPHA_MODEL_PATH = os.path.join(_REPO_ROOT, "models", "psl", "psl_landmark_classifier.tflite")
ALPHA_ENCODER_PATH = os.path.join(_REPO_ROOT, "models", "psl", "label_encoder.pkl")
ALPHA_SCALER_PATH = os.path.join(_REPO_ROOT, "models", "psl", "scaler.pkl")


# ── Module-level model singletons ──────────────────────────────────────
_alpha_lock = threading.Lock()
_alpha_interpreter: Optional[tf.lite.Interpreter] = None
_alpha_in_idx: int = 0
_alpha_out_idx: int = 0
_alpha_label_encoder = None
_alpha_scaler = None
_alpha_classes: list[str] = []


def _ensure_alpha_model_loaded() -> None:
    global _alpha_interpreter, _alpha_in_idx, _alpha_out_idx
    global _alpha_label_encoder, _alpha_scaler, _alpha_classes
    if _alpha_interpreter is not None:
        return
    for p in (ALPHA_MODEL_PATH, ALPHA_ENCODER_PATH, ALPHA_SCALER_PATH):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Alphabet asset missing: {p}")
    interp = tf.lite.Interpreter(model_path=ALPHA_MODEL_PATH)
    interp.allocate_tensors()
    in_det = interp.get_input_details()
    out_det = interp.get_output_details()
    _alpha_interpreter = interp
    _alpha_in_idx = in_det[0]["index"]
    _alpha_out_idx = out_det[0]["index"]
    _alpha_label_encoder = joblib.load(ALPHA_ENCODER_PATH)
    _alpha_scaler = joblib.load(ALPHA_SCALER_PATH)
    _alpha_classes = list(_alpha_label_encoder.classes_)
    print(f"[server] alphabet model loaded: {len(_alpha_classes)} classes")


# ── FSM (mirrors SignSession's outward state vocabulary) ───────────────
class AlphaState(Enum):
    IDLE = auto()
    SIGNING = auto()
    COOLDOWN = auto()


_STATE_LABEL = {
    AlphaState.IDLE: "IDLE",
    AlphaState.SIGNING: "SIGNING",
    AlphaState.COOLDOWN: "COOLDOWN",
}


class AlphabetSession:
    """One per WebSocket connection. Not thread-safe — call from one task."""

    def __init__(self, conf_threshold: float = THRESHOLD,
                 stable_required: int = STABLE_REQUIRED) -> None:
        _ensure_alpha_model_loaded()
        self.conf_threshold = conf_threshold
        self.stable_required = stable_required

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self._state = AlphaState.IDLE
        self._cooldown_until = 0.0

        self._last_label = ""
        self._last_conf = 0.0
        self._stable_count = 0
        self._committed_this_frame = False

    def close(self) -> None:
        try:
            self._hands.close()
        except Exception:
            pass

    def process_jpeg(self, jpeg_bytes: bytes) -> dict:
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return self._snapshot(has_hands=False, error="bad_jpeg")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)
        has_hands = bool(results.multi_hand_landmarks)

        now = time.time()
        self._committed_this_frame = False

        if self._state is AlphaState.COOLDOWN:
            if now >= self._cooldown_until:
                self._state = AlphaState.IDLE
                self._stable_count = 0

        if not has_hands:
            if self._state is AlphaState.SIGNING:
                self._state = AlphaState.IDLE
                self._stable_count = 0
            return self._snapshot(has_hands=False)

        # Extract 42-D vec (x,y) of first hand.
        lms = results.multi_hand_landmarks[0].landmark
        vec = np.array([[p.x, p.y] for p in lms], dtype=np.float32).reshape(1, -1)

        with _alpha_lock:
            scaled = _alpha_scaler.transform(vec).astype(np.float32)
            _alpha_interpreter.set_tensor(_alpha_in_idx, scaled)
            _alpha_interpreter.invoke()
            probs = _alpha_interpreter.get_tensor(_alpha_out_idx)[0]

        top = int(np.argmax(probs))
        conf = float(probs[top])
        label = _alpha_classes[top] if 0 <= top < len(_alpha_classes) else ""

        self._last_label = label
        self._last_conf = conf

        if self._state is AlphaState.COOLDOWN:
            return self._snapshot(has_hands=True)

        self._state = AlphaState.SIGNING

        if conf >= self.conf_threshold:
            if label == getattr(self, "_prev_stable_label", None):
                self._stable_count += 1
            else:
                self._prev_stable_label = label
                self._stable_count = 1

            if self._stable_count >= self.stable_required:
                self._committed_this_frame = True
                self._stable_count = 0
                self._cooldown_until = now + COOLDOWN_SECONDS
                self._state = AlphaState.COOLDOWN
        else:
            self._stable_count = 0
            self._prev_stable_label = None

        return self._snapshot(has_hands=True)

    def _snapshot(self, has_hands: bool, error: Optional[str] = None) -> dict:
        urdu = PSL_TO_URDU.get(self._last_label, self._last_label)
        # Stability ratio mapped onto bufferFill / bufferCapacity so the
        # mobile UI can re-use the existing progress bar widget.
        return {
            "state": _STATE_LABEL[self._state],
            "label": self._last_label,
            "english": self._last_label,
            "urdu": urdu,
            "confidence": float(self._last_conf),
            "committed": bool(self._committed_this_frame),
            "hasHands": bool(has_hands),
            "bufferFill": int(self._stable_count),
            "bufferCapacity": int(self.stable_required),
            "error": error,
        }
