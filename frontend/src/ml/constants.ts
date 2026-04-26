// ML pipeline constants — ported from src/inference/psl_words_v2.py
// DO NOT change these values without re-testing against the model.

// ---- Model contract ------------------------------------------------
export const T_WINDOW = 60;            // rolling window length
export const F_DIM = 126;              // per-frame feature dim
export const HAND_COUNT = 2;
export const LANDMARKS_PER_HAND = 21;
export const COORDS_PER_LANDMARK = 3;
export const NUM_CLASSES = 64;

// ---- FSM / thresholds ----------------------------------------------
export const STRIDE_FRAMES = 3;
export const COMMIT_CONF_DEFAULT = 0.70;
export const CONF_MIN = 0.40;
export const CONF_MAX = 0.95;
export const COOLDOWN_SECONDS = 0.8;
export const UNDO_HOLD_SECONDS = 2.0;
export const MOTION_VAR_MIN = 1e-4;

// ---- Smoothing -----------------------------------------------------
export const EMA_DISPLAY_ALPHA = 0.30;
export const EMA_INFER_ALPHA = 0.60;

// ---- Classes to ignore as commits ----------------------------------
export const IDLE_CLASSES = new Set(["nothing", "test_word"]);

// ---- Handedness inversion -------------------------------------------
// Desktop Python sets HANDS_INVERT_HANDEDNESS=True because frames are
// NOT selfie-flipped before being sent to MediaPipe. The same applies
// when using vision-camera frame processors on the front camera.
export const HANDS_INVERT_HANDEDNESS = true;
