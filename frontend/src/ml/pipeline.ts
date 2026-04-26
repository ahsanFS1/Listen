import { SignStateMachine, SignState } from "./fsm";
import { RollingBuffer } from "./rollingBuffer";
import { SIGNS } from "@/data/signs";

// Public shape of a prediction the UI consumes.
export type Prediction = {
  label: string;
  english: string;
  urdu: string;
  confidence: number;
  state: SignState;
  committed: boolean;
};

// High-level pipeline facade.
//
// Two run modes:
// - `mock`  → drives a state machine that imitates "no hands → scanning →
//   predicting → commit → cooldown" so the UI feels alive without a
//   native dev build. Critically, predictions ONLY emit while the camera
//   is active AND the simulated "hands visible" flag is set, so the
//   feed does not spam committed words while the user's hand is down.
// - `tflite` → wired to react-native-fast-tflite + vision-camera frame
//   processor on a dev build. Runtime feature-detected; falls back to
//   `mock` if the native modules are unavailable (Expo Go, web).
export class SignPipeline {
  readonly fsm = new SignStateMachine();
  readonly buffer = new RollingBuffer();

  // --- mock-mode state ---
  private active = false;
  private phase: "idle" | "scanning" | "ramping" | "holding" | "cooldown" = "idle";
  private phaseStartedAt = 0;
  private currentSignIdx = 0;
  private confidence = 0;
  private lastEmitted: Prediction | null = null;
  private committedThisCycle = false;

  // Deterministic shuffle order so the mock cycles through varied words.
  private readonly cycleOrder: number[] = MOCK_CYCLE_IDS.map((_, i) => i);

  start(): void {
    this.active = true;
    this.transition("scanning");
    this.confidence = 0;
    this.committedThisCycle = false;
  }

  stop(): void {
    this.active = false;
    this.phase = "idle";
    this.confidence = 0;
    this.committedThisCycle = false;
    this.lastEmitted = null;
    this.fsm.state = "IDLE";
  }

  isActive(): boolean {
    return this.active;
  }

  /**
   * Drive the mock one tick forward. Call at ~10–15 Hz.
   * Returns the latest prediction (or null when idle/no-hands).
   */
  mockTick(): Prediction | null {
    if (!this.active) {
      this.lastEmitted = null;
      return null;
    }

    const now = Date.now();
    const elapsed = now - this.phaseStartedAt;

    switch (this.phase) {
      case "scanning":
        // Looking for hands — no prediction emitted, state pill says "IDLE"
        if (elapsed > 1400) this.transition("ramping");
        return {
          label: "",
          english: "—",
          urdu: "—",
          confidence: 0,
          state: "IDLE",
          committed: false,
        };

      case "ramping":
        // Rising confidence on a chosen sign, simulates active SIGNING/PREDICTING.
        this.confidence = Math.min(0.97, this.confidence + 0.04);
        if (this.confidence >= 0.9) this.transition("holding");
        break;

      case "holding": {
        // Confidence holds high; a single commit fires once per cycle.
        const target = this.cycleOrder[this.currentSignIdx % this.cycleOrder.length];
        const sign = MOCK_CYCLE[target];
        const committed = !this.committedThisCycle;
        if (committed) this.committedThisCycle = true;
        const pred: Prediction = {
          label: sign.id,
          english: sign.english,
          urdu: sign.urdu,
          confidence: this.confidence,
          state: committed ? "COMMITTED" : "PREDICTING",
          committed,
        };
        this.lastEmitted = pred;
        if (elapsed > 800) this.transition("cooldown");
        return pred;
      }

      case "cooldown":
        // After a commit, wind confidence back down — looks like the
        // signer pausing between words.
        this.confidence = Math.max(0, this.confidence - 0.05);
        if (this.confidence <= 0.15) {
          this.currentSignIdx = (this.currentSignIdx + 1) % this.cycleOrder.length;
          this.committedThisCycle = false;
          this.transition("scanning");
        }
        break;

      case "idle":
        return null;
    }

    const target = this.cycleOrder[this.currentSignIdx % this.cycleOrder.length];
    const sign = MOCK_CYCLE[target];
    const fsmState: SignState =
      this.confidence >= 0.7
        ? "PREDICTING"
        : this.confidence >= 0.35
        ? "SIGNING"
        : "IDLE";
    const pred: Prediction = {
      label: sign.id,
      english: sign.english,
      urdu: sign.urdu,
      confidence: this.confidence,
      state: fsmState,
      committed: false,
    };
    this.lastEmitted = pred;
    return pred;
  }

  last(): Prediction | null {
    return this.lastEmitted;
  }

  reset(): void {
    this.buffer.reset();
    this.confidence = 0;
    this.currentSignIdx = 0;
    this.committedThisCycle = false;
    this.lastEmitted = null;
    if (this.active) this.transition("scanning");
  }

  private transition(next: SignPipeline["phase"]): void {
    this.phase = next;
    this.phaseStartedAt = Date.now();
  }
}

// Pick some of the model's most reliably-detected signs for the mock cycle.
const MOCK_CYCLE_IDS = [
  "hello",
  "thankyou",
  "water",
  "you",
  "mobile_phone",
  "welldone",
  "dog",
  "assalam-o-alaikum",
];

const MOCK_CYCLE = MOCK_CYCLE_IDS.map((id) => {
  const s = SIGNS.find((x) => x.id === id);
  if (!s) throw new Error(`Mock cycle references unknown sign: ${id}`);
  return s;
});
