import {
  COORDS_PER_LANDMARK,
  F_DIM,
  HANDS_INVERT_HANDEDNESS,
  HAND_COUNT,
  LANDMARKS_PER_HAND,
} from "./constants";

// A single MediaPipe landmark (x, y, z in normalized coords).
export type Landmark = { x: number; y: number; z: number };

// Output of the MediaPipe hand detector for a single frame.
export type HandDetection = {
  // "Left" or "Right" from MediaPipe's classifier.
  // Because the frame is NOT selfie-flipped, "Right" typically means the
  // signer's anatomical LEFT hand. See HANDS_INVERT_HANDEDNESS.
  handedness: "Left" | "Right";
  landmarks: Landmark[]; // must be length 21
};

// Build the 126-D per-frame feature vector.
// Layout: [lh_x0,lh_y0,lh_z0,...,lh_x20,lh_y20,lh_z20, rh_x0,...,rh_z20]
// Matches `frame_from_mediapipe` in psl_words_v2.py.
export function extractFrame(detections: HandDetection[]): Float32Array {
  const frame = new Float32Array(F_DIM);
  // Track which hands we've filled to avoid a second detection overwriting
  // the first (matches `if self.left_hand_landmarks is None:` check).
  let leftFilled = false;
  let rightFilled = false;

  for (const det of detections) {
    const isAnatomicalLeft = HANDS_INVERT_HANDEDNESS
      ? det.handedness === "Right"
      : det.handedness === "Left";
    const offset = isAnatomicalLeft ? 0 : LANDMARKS_PER_HAND * COORDS_PER_LANDMARK;

    if (isAnatomicalLeft && leftFilled) continue;
    if (!isAnatomicalLeft && rightFilled) continue;

    if (det.landmarks.length < LANDMARKS_PER_HAND) continue;
    for (let j = 0; j < LANDMARKS_PER_HAND; j++) {
      const lm = det.landmarks[j];
      const base = offset + j * COORDS_PER_LANDMARK;
      frame[base + 0] = lm.x;
      frame[base + 1] = lm.y;
      frame[base + 2] = lm.z;
    }
    if (isAnatomicalLeft) leftFilled = true;
    else rightFilled = true;
  }
  return frame;
}

// Per-hand wrist-centred, max-abs normalization.
// Matches `normalize_frame` in psl_words_v2.py.
export function normalizeFrame(frame: Float32Array): Float32Array {
  const out = new Float32Array(frame);
  for (let h = 0; h < HAND_COUNT; h++) {
    const base = h * LANDMARKS_PER_HAND * COORDS_PER_LANDMARK;
    // Detect a completely absent hand (all zeros) — leave it as zeros.
    let anyNonZero = false;
    for (let k = base; k < base + LANDMARKS_PER_HAND * COORDS_PER_LANDMARK; k++) {
      if (out[k] !== 0) {
        anyNonZero = true;
        break;
      }
    }
    if (!anyNonZero) continue;

    const wx = out[base + 0];
    const wy = out[base + 1];
    const wz = out[base + 2];
    for (let j = 0; j < LANDMARKS_PER_HAND; j++) {
      const idx = base + j * COORDS_PER_LANDMARK;
      out[idx + 0] -= wx;
      out[idx + 1] -= wy;
      out[idx + 2] -= wz;
    }

    let maxAbs = 0;
    for (let k = base; k < base + LANDMARKS_PER_HAND * COORDS_PER_LANDMARK; k++) {
      const a = Math.abs(out[k]);
      if (a > maxAbs) maxAbs = a;
    }
    if (maxAbs > 0) {
      const inv = 1 / maxAbs;
      for (let k = base; k < base + LANDMARKS_PER_HAND * COORDS_PER_LANDMARK; k++) {
        out[k] *= inv;
      }
    }
  }
  return out;
}

// Mean-squared difference between two normalized frames.
// Used for the motion gate in the FSM (MOTION_VAR_MIN threshold).
export function frameMotion(
  a: Float32Array | null,
  b: Float32Array | null,
): number {
  if (!a || !b) return 0;
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return sum / a.length;
}
