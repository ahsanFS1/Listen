// Real TFLite inference for the words model.
// Loads `assets/models/psl_word_classifier.tflite` via react-native-fast-tflite.
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
};

export const isTfliteAvailable = (): boolean => loadFastTflite() != null;

let _wordsModel: LoadedModel | null = null;
let _loading: Promise<LoadedModel | null> | null = null;

// Load the bundled words model. Returns null if the native module isn't
// present (we then fall back to mock).
export async function loadWordsModel(): Promise<LoadedModel | null> {
  if (_wordsModel) return _wordsModel;
  if (_loading) return _loading;
  const mod = loadFastTflite();
  if (!mod || !mod.loadTensorflowModel) return null;

  _loading = (async () => {
    try {
      // The path resolves through Metro's `require` like any other asset.
      const model = await mod.loadTensorflowModel(
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        require("../../assets/models/psl_word_classifier.tflite"),
      );
      _wordsModel = {
        model,
        inputShape: [1, T_WINDOW, F_DIM],
        outputShape: [1, NUM_CLASSES],
      };
      return _wordsModel;
    } catch (e) {
      console.warn("[tfliteRunner] failed to load words model", e);
      return null;
    } finally {
      _loading = null;
    }
  })();
  return _loading;
}

// Run a single inference pass on a flat (T_WINDOW * F_DIM) Float32Array.
// Returns a probability vector of length NUM_CLASSES, or null on failure.
export async function classifyWindow(
  window: Float32Array,
): Promise<Float32Array | null> {
  const loaded = await loadWordsModel();
  if (!loaded) return null;
  try {
    const out = await loaded.model.run([window]);
    if (!out || !out[0]) return null;
    const arr = out[0] as Float32Array | number[];
    return arr instanceof Float32Array ? arr : new Float32Array(arr);
  } catch (e) {
    console.warn("[tfliteRunner] inference failed", e);
    return null;
  }
}
