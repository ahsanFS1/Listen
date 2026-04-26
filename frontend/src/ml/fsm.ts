import {
  COMMIT_CONF_DEFAULT,
  COOLDOWN_SECONDS,
  IDLE_CLASSES,
  MOTION_VAR_MIN,
  STRIDE_FRAMES,
  T_WINDOW,
} from "./constants";

export type SignState =
  | "IDLE"
  | "SIGNING"
  | "PREDICTING"
  | "COMMITTED"
  | "COOLDOWN";

// Finite-state-machine for the sign recognition pipeline.
// Ported from ListenApp run loop in psl_words_v2.py (lines ~1499-1529).
export class SignStateMachine {
  state: SignState = "IDLE";
  private stateSince = Date.now();
  private cooldownUntil = 0;
  private framesSinceInfer = 0;
  private lastCommittedLabel: string | null = null;

  commitThreshold = COMMIT_CONF_DEFAULT;

  tick(params: {
    hasHands: boolean;
    motion: number;
    bufferFull: boolean;
  }): void {
    const { hasHands, motion, bufferFull } = params;
    const now = Date.now();
    this.framesSinceInfer++;

    switch (this.state) {
      case "IDLE":
        if (hasHands && motion > MOTION_VAR_MIN) {
          this.enter("SIGNING");
        }
        break;

      case "SIGNING":
        if (
          !hasHands &&
          motion < MOTION_VAR_MIN &&
          now - this.stateSince > 1200
        ) {
          this.enter("IDLE");
        }
        break;

      case "COMMITTED":
        this.enter("COOLDOWN");
        break;

      case "COOLDOWN":
        if (hasHands) this.cooldownUntil = now + COOLDOWN_SECONDS * 1000;
        if (now >= this.cooldownUntil && !hasHands) this.enter("IDLE");
        break;

      case "PREDICTING":
        // External caller triggers this via onPrediction()
        break;
    }
    // Suppress unused warning for bufferFull; used by shouldInfer().
    void bufferFull;
  }

  shouldInfer(motion: number, bufferFull: boolean): boolean {
    if (!bufferFull) return false;
    if (this.framesSinceInfer < STRIDE_FRAMES) return false;
    if (motion < MOTION_VAR_MIN) return false;
    if (this.state !== "SIGNING") return false;
    return true;
  }

  markInferSubmitted(): void {
    this.framesSinceInfer = 0;
    this.enter("PREDICTING");
  }

  onPrediction(label: string, confidence: number): {
    committed: boolean;
    label: string;
  } {
    if (this.state !== "PREDICTING") return { committed: false, label };
    if (!IDLE_CLASSES.has(label) && confidence >= this.commitThreshold) {
      this.lastCommittedLabel = label;
      this.enter("COMMITTED");
      return { committed: true, label };
    }
    this.enter("SIGNING");
    return { committed: false, label };
  }

  private enter(state: SignState): void {
    if (state === this.state) return;
    this.state = state;
    this.stateSince = Date.now();
    if (state === "COOLDOWN") {
      this.cooldownUntil = this.stateSince + COOLDOWN_SECONDS * 1000;
      this.framesSinceInfer = 0;
    }
    if (state === "IDLE") {
      this.framesSinceInfer = 0;
    }
  }

  stateLabel(): string {
    return this.state;
  }
}

// EMA smoothing for the probability vector (infer side) and the display
// confidence scalar (UI side). Matches EMA_INFER_ALPHA / EMA_DISPLAY_ALPHA.
export function emaVector(
  prev: Float32Array | null,
  next: Float32Array,
  alpha: number,
): Float32Array {
  if (!prev) return new Float32Array(next);
  const out = new Float32Array(next.length);
  for (let i = 0; i < next.length; i++) {
    out[i] = alpha * prev[i] + (1 - alpha) * next[i];
  }
  return out;
}

export function emaScalar(prev: number, next: number, alpha: number): number {
  return alpha * next + (1 - alpha) * prev;
}

export { T_WINDOW };
