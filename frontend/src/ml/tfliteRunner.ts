// Real TFLite inference for words and alphabet models.
// Loads models via react-native-fast-tflite.
//
// IMPORTANT: react-native-fast-tflite is a NATIVE module. It only works in a
// dev build (`npx expo run:android` or run:ios), NOT in Expo Go or on web.
// To keep Expo Go usable for UI iteration, every entry point here is
// feature-detected — if the native module is unavailable we set
// `available = false` and the calling code falls back to the mock pipeline.

import { F_DIM, NUM_CLASSES, T_WINDOW } from "./constants";

// Lazy-load the module so importing this file in Expo Go doesn't crash.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let _fastTflite: any = null;
let _loadAttempted = false;

function loadFastTflite() {
  if (_loadAttempted) return _fastTflite;
  _loadAttempted = true;
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    _fastTflite = require("react-native-fast-tflite");
  } catch {
    _fastTflite = null;
  }
  return _fastTflite;
}

export type LoadedModel = {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  model: any;
  inputShape: number[];
  outputShape: number[];
  numClasses: number;
};

export type ModelType = "words" | "alphabet";

export const isTfliteAvailable = (): boolean => loadFastTflite() != null;

let _wordsModel: LoadedModel | null = null;
let _alphabetModel: LoadedModel | null = null;
let _wordsLoading: Promise<LoadedModel | null> | null = null;
let _alphabetLoading: Promise<LoadedModel | null> | null = null;

// Alphabet model constants
const ALPHABET_F_DIM = 126; // Same landmark features
const ALPHABET_NUM_CLASSES = 40; // 40 Urdu alphabets

// Load the bundled words model. Returns null if the native module isn't
// present (we then fall back to mock).
export async function loadWordsModel(): Promise<LoadedModel | null> {
  if (_wordsModel) return _wordsModel;
  if (_wordsLoading) return _wordsLoading;
  const mod = loadFastTflite();
  if (!mod || !mod.loadTensorflowModel) return null;

  _wordsLoading = (async () => {
    try {
      const model = await mod.loadTensorflowModel(
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        require("../../assets/models/psl_word_classifier.tflite"),
      );
      _wordsModel = {
        model,
        inputShape: [1, T_WINDOW, F_DIM],
        outputShape: [1, NUM_CLASSES],
        numClasses: NUM_CLASSES,
      };
      console.log("[tfliteRunner] Words model loaded successfully");
      return _wordsModel;
    } catch (e) {
      console.warn("[tfliteRunner] failed to load words model", e);
      return null;
    } finally {
      _wordsLoading = null;
    }
  })();
  return _wordsLoading;
}

// Load the bundled alphabet model
export async function loadAlphabetModel(): Promise<LoadedModel | null> {
  if (_alphabetModel) return _alphabetModel;
  if (_alphabetLoading) return _alphabetLoading;
  const mod = loadFastTflite();
  if (!mod || !mod.loadTensorflowModel) return null;

  _alphabetLoading = (async () => {
    try {
      const model = await mod.loadTensorflowModel(
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        require("../../assets/models/psl_alphabet_classifier.tflite"),
      );
      _alphabetModel = {
        model,
        inputShape: [1, T_WINDOW, ALPHABET_F_DIM],
        outputShape: [1, ALPHABET_NUM_CLASSES],
        numClasses: ALPHABET_NUM_CLASSES,
      };
      console.log("[tfliteRunner] Alphabet model loaded successfully");
      return _alphabetModel;
    } catch (e) {
      console.warn("[tfliteRunner] failed to load alphabet model", e);
      return null;
    } finally {
      _alphabetLoading = null;
    }
  })();
  return _alphabetLoading;
}

// Load model by type
export async function loadModel(type: ModelType): Promise<LoadedModel | null> {
  return type === "words" ? loadWordsModel() : loadAlphabetModel();
}

// Run a single inference pass on a flat (T_WINDOW * F_DIM) Float32Array.
// Returns a probability vector, or null on failure.
export async function classifyWindow(
  window: Float32Array,
  modelType: ModelType = "words",
): Promise<Float32Array | null> {
  const loaded = await loadModel(modelType);
  if (!loaded) return null;
  try {
    const out = await loaded.model.run([window]);
    if (!out || !out[0]) return null;
    const arr = out[0] as Float32Array | number[];
    return arr instanceof Float32Array ? arr : new Float32Array(arr);
  } catch (e) {
    console.warn(`[tfliteRunner] ${modelType} inference failed`, e);
    return null;
  }
}

// Get top prediction from probability vector
export function getTopPrediction(
  probs: Float32Array,
  labels: string[],
): { label: string; confidence: number; index: number } | null {
  if (!probs || probs.length === 0) return null;
  
  let maxIdx = 0;
  let maxProb = probs[0];
  
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > maxProb) {
      maxProb = probs[i];
      maxIdx = i;
    }
  }
  
  return {
    label: labels[maxIdx] || "unknown",
    confidence: maxProb,
    index: maxIdx,
  };
}
