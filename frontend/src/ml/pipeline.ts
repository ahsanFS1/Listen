import { SignStateMachine, SignState } from "./fsm";
import { RollingBuffer } from "./rollingBuffer";
import { classifyWindow, getTopPrediction, ModelType } from "./tfliteRunner";
import { extractFrame, normalizeFrame, frameMotion, HandDetection } from "./featureExtractor";
import { SIGNS } from "@/data/signs";

// Sign data type
interface SignData {
  id: string;
  english: string;
  urdu: string;
  category: string;
}

// 40 Urdu alphabet signs for PSL
const ALPHABET_SIGNS: SignData[] = [
  { id: "alif", english: "Alif", urdu: "ا", category: "alphabet" },
  { id: "bay", english: "Bay", urdu: "ب", category: "alphabet" },
  { id: "pay", english: "Pay", urdu: "پ", category: "alphabet" },
  { id: "tay", english: "Tay", urdu: "ت", category: "alphabet" },
  { id: "ttaay", english: "Ttaay", urdu: "ٹ", category: "alphabet" },
  { id: "say", english: "Say", urdu: "ث", category: "alphabet" },
  { id: "jeem", english: "Jeem", urdu: "ج", category: "alphabet" },
  { id: "chay", english: "Chay", urdu: "چ", category: "alphabet" },
  { id: "hay", english: "Hay", urdu: "ح", category: "alphabet" },
  { id: "khey", english: "Khey", urdu: "خ", category: "alphabet" },
  { id: "daal", english: "Daal", urdu: "د", category: "alphabet" },
  { id: "daal_d", english: "Daal (Heavy)", urdu: "ڈ", category: "alphabet" },
  { id: "zaal", english: "Zaal", urdu: "ذ", category: "alphabet" },
  { id: "ray", english: "Ray", urdu: "ر", category: "alphabet" },
  { id: "rey", english: "Rey (Heavy)", urdu: "ڑ", category: "alphabet" },
  { id: "zay", english: "Zay", urdu: "ز", category: "alphabet" },
  { id: "zhay", english: "Zhay", urdu: "ژ", category: "alphabet" },
  { id: "seen", english: "Seen", urdu: "س", category: "alphabet" },
  { id: "sheen", english: "Sheen", urdu: "ش", category: "alphabet" },
  { id: "suad", english: "Suad", urdu: "ص", category: "alphabet" },
  { id: "zuad", english: "Zuad", urdu: "ض", category: "alphabet" },
  { id: "toy", english: "Toy", urdu: "ط", category: "alphabet" },
  { id: "zoy", english: "Zoy", urdu: "ظ", category: "alphabet" },
  { id: "ain", english: "Ain", urdu: "ع", category: "alphabet" },
  { id: "ghain", english: "Ghain", urdu: "غ", category: "alphabet" },
  { id: "fay", english: "Fay", urdu: "ف", category: "alphabet" },
  { id: "qaf", english: "Qaf", urdu: "ق", category: "alphabet" },
  { id: "kaf", english: "Kaf", urdu: "ک", category: "alphabet" },
  { id: "gaf", english: "Gaf", urdu: "گ", category: "alphabet" },
  { id: "laam", english: "Laam", urdu: "ل", category: "alphabet" },
  { id: "meem", english: "Meem", urdu: "م", category: "alphabet" },
  { id: "noon", english: "Noon", urdu: "ن", category: "alphabet" },
  { id: "noon_g", english: "Noon Ghunna", urdu: "ں", category: "alphabet" },
  { id: "wao", english: "Wao", urdu: "و", category: "alphabet" },
  { id: "wao_h", english: "Wao (Heavy)", urdu: "ؤ", category: "alphabet" },
  { id: "hay_h", english: "Hay Do Ashar", urdu: "ہ", category: "alphabet" },
  { id: "choti_yay", english: "Choti Yay", urdu: "ی", category: "alphabet" },
  { id: "badi_yay", english: "Badi Yay", urdu: "ے", category: "alphabet" },
  { id: "hamza", english: "Hamza", urdu: "ء", category: "alphabet" },
  { id: "chay_h", english: "Chay Do Ashar", urdu: "چ", category: "alphabet" },
];

// Public shape of a prediction the UI consumes.
export type Prediction = {
  label: string;
  english: string;
  urdu: string;
  confidence: number;
  state: SignState;
  committed: boolean;
  hasHands: boolean;
};

export type PipelineMode = "words" | "alphabet";

// High-level pipeline facade with real hand detection.
// Uses MediaPipe hand landmarks and TFLite models for on-device inference.
export class SignPipeline {
  readonly fsm = new SignStateMachine();
  readonly buffer = new RollingBuffer();
  
  mode: PipelineMode = "words";
  
  // --- state ---
  private active = false;
  private lastEmitted: Prediction | null = null;
  private lastFrame: Float32Array | null = null;
  private handsVisible = false;
  private currentMotion = 0;
  
  // For EMA smoothing
  private emaProbs: Float32Array | null = null;
  private readonly EMA_ALPHA = 0.6;

  // Labels cache
  private wordLabels: string[] = [];
  private alphabetLabels: string[] = [];

  // Mock hold-session state: pick one sign per press, ramp it, commit once.
  private mockSessionStart: number | null = null;
  private mockSessionLabel: string | null = null;
  private mockSessionCommitted = false;

  constructor() {
    // Load labels
    this.loadLabels();
  }

  private async loadLabels() {
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const wordData = require("../../assets/models/class_labels.json");
      this.wordLabels = wordData;
      
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const alphabetData = require("../../assets/models/alphabet_labels.json");
      this.alphabetLabels = alphabetData;
    } catch (e) {
      console.warn("[SignPipeline] Failed to load labels", e);
      // Fallback to SIGNS
      this.wordLabels = SIGNS.map((s: SignData) => s.id);
      this.alphabetLabels = ALPHABET_SIGNS.map((s: SignData) => s.id);
    }
  }

  setMode(mode: PipelineMode) {
    this.mode = mode;
    this.reset();
  }

  start(): void {
    this.active = true;
    this.fsm.state = "IDLE";
    console.log(`[SignPipeline] Started in ${this.mode} mode`);
  }

  stop(): void {
    this.active = false;
    this.handsVisible = false;
    this.lastEmitted = null;
    this.fsm.state = "IDLE";
  }

  isActive(): boolean {
    return this.active;
  }

  /**
   * Process hand detections from camera frame.
   * Call this from the frame processor at ~15-20 FPS.
   */
  async processFrame(detections: HandDetection[]): Promise<Prediction | null> {
    if (!this.active) {
      this.lastEmitted = null;
      return null;
    }

    // Check if hands are visible
    this.handsVisible = detections.length > 0 && 
      detections.some(d => d.landmarks && d.landmarks.length >= 21);

    // If no hands visible, return idle state immediately
    if (!this.handsVisible) {
      this.buffer.reset();
      this.lastFrame = null;
      this.emaProbs = null;
      this.fsm.state = "IDLE";
      const idlePred: Prediction = {
        label: "",
        english: "—",
        urdu: "—",
        confidence: 0,
        state: "IDLE",
        committed: false,
        hasHands: false,
      };
      this.lastEmitted = idlePred;
      return idlePred;
    }

    // Extract and normalize features
    const frame = extractFrame(detections);
    const normalized = normalizeFrame(frame);
    
    // Calculate motion
    this.currentMotion = frameMotion(this.lastFrame, normalized);
    this.lastFrame = normalized;
    
    // Push to buffer
    this.buffer.push(normalized);
    
    // Update FSM
    this.fsm.tick({
      hasHands: this.handsVisible,
      motion: this.currentMotion,
      bufferFull: this.buffer.isFull,
    });

    // Should we run inference?
    if (this.fsm.shouldInfer(this.currentMotion, this.buffer.isFull)) {
      const window = this.buffer.snapshot();
      const probs = await classifyWindow(window, this.mode);
      
      if (probs) {
        // EMA smoothing
        if (this.emaProbs) {
          for (let i = 0; i < probs.length; i++) {
            probs[i] = this.EMA_ALPHA * probs[i] + (1 - this.EMA_ALPHA) * this.emaProbs[i];
          }
        }
        this.emaProbs = new Float32Array(probs);
        
        // Get prediction
        const labels = this.mode === "words" ? this.wordLabels : this.alphabetLabels;
        const topPred = getTopPrediction(probs, labels);
        
        if (topPred) {
          const sign = this.getSignData(topPred.label);
          const commitResult = this.fsm.onPrediction(topPred.label, topPred.confidence);
          
          const pred: Prediction = {
            label: topPred.label,
            english: sign.english,
            urdu: sign.urdu,
            confidence: topPred.confidence,
            state: commitResult.committed ? "COMMITTED" : "PREDICTING",
            committed: commitResult.committed,
            hasHands: true,
          };
          
          this.fsm.markInferSubmitted();
          this.lastEmitted = pred;
          return pred;
        }
      }
    }

    // Return current state without new prediction
    const currentSign = this.lastEmitted || { label: "", english: "—", urdu: "—" };
    const state: Prediction = {
      ...currentSign,
      confidence: this.lastEmitted?.confidence || 0,
      state: this.fsm.state,
      committed: false,
      hasHands: true,
    };
    this.lastEmitted = state;
    return state;
  }

  /**
   * Mock tick for testing without camera - respects hand visibility simulation
   */
  mockTick(simulateHands = true): Prediction | null {
    if (!this.active) {
      this.lastEmitted = null;
      return null;
    }

    if (!simulateHands) {
      // Button released — end any active session and go idle.
      this.handsVisible = false;
      this.buffer.reset();
      this.emaProbs = null;
      this.fsm.state = "IDLE";
      this.mockSessionStart = null;
      this.mockSessionLabel = null;
      this.mockSessionCommitted = false;
      const idlePred: Prediction = {
        label: "",
        english: "—",
        urdu: "—",
        confidence: 0,
        state: "IDLE",
        committed: false,
        hasHands: false,
      };
      this.lastEmitted = idlePred;
      return idlePred;
    }

    this.handsVisible = true;

    const labels = this.mode === "words" ? this.wordLabels : this.alphabetLabels;
    if (labels.length === 0) {
      return {
        label: "",
        english: "—",
        urdu: "—",
        confidence: 0,
        state: "IDLE",
        committed: false,
        hasHands: true,
      };
    }

    // Start a hold-session on first tick: pick one random sign and stick with it
    // until the user releases. This avoids the previous "cycle through every label"
    // behavior that made the UI look like it was spamming words.
    const now = Date.now();
    if (this.mockSessionStart === null || this.mockSessionLabel === null) {
      this.mockSessionStart = now;
      this.mockSessionLabel = labels[Math.floor(Math.random() * labels.length)];
      this.mockSessionCommitted = false;
    }

    const elapsed = now - this.mockSessionStart;
    const label = this.mockSessionLabel;
    const sign = this.getSignData(label);

    // Confidence curve over 1.5s ramp, then hold; commit once at ~1.0s.
    let confidence = 0;
    let state: SignState = "SIGNING";
    if (elapsed < 500) {
      confidence = (elapsed / 500) * 0.4;
      state = "SIGNING";
    } else if (elapsed < 1000) {
      confidence = 0.4 + ((elapsed - 500) / 500) * 0.55;
      state = "PREDICTING";
    } else {
      confidence = 0.95;
      state = "PREDICTING";
    }

    let committed = false;
    if (!this.mockSessionCommitted && elapsed >= 1000) {
      committed = true;
      this.mockSessionCommitted = true;
    }

    const pred: Prediction = {
      label,
      english: sign.english,
      urdu: sign.urdu,
      confidence,
      state: committed ? "COMMITTED" : state,
      committed,
      hasHands: true,
    };

    this.lastEmitted = pred;
    return pred;
  }

  last(): Prediction | null {
    return this.lastEmitted;
  }

  reset(): void {
    this.buffer.reset();
    this.lastFrame = null;
    this.emaProbs = null;
    this.handsVisible = false;
    this.lastEmitted = null;
    this.fsm.state = "IDLE";
    this.mockSessionStart = null;
    this.mockSessionLabel = null;
    this.mockSessionCommitted = false;
  }

  private getSignData(label: string): { english: string; urdu: string } {
    const signs = this.mode === "words" ? SIGNS : ALPHABET_SIGNS;
    const sign = signs.find(s => s.id === label);
    if (sign) return { english: sign.english, urdu: sign.urdu };
    
    // Fallback for unknown labels
    return { 
      english: label.charAt(0).toUpperCase() + label.slice(1).replace(/_/g, " "),
      urdu: label,
    };
  }
}

// For backward compatibility - these will be removed once real ML is fully wired
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
