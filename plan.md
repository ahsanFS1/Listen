# Word-Level PSL Recognition — Plan

## 1. Current State of the Codebase (Alphabet-Level)

The existing pipeline is a **per-frame, static-image Urdu alphabet classifier**:

| Stage | File | What it does |
|------|------|--------------|
| Landmark extraction | `src/preprocessing/extract_landmarks_psl.py` | Runs MediaPipe Hands on each image in `UAlpha40…` and writes one row per image to `data/psl_landmarks.csv`. Features = 21 landmarks × (x, y) = **42 values**, single hand. |
| Preprocess | `src/preprocessing/preprocess_landmarks_psl.py` | `StandardScaler` + `LabelEncoder`, 80/20 stratified split, saves to `data/psl_processed/`. |
| Train | `src/training/train_psl.py` | A small **MLP** (`42 → 256 → 128 → num_classes`) trained with `sparse_categorical_crossentropy`. Exports `.h5` and `.tflite`. |
| Inference | `src/inference/psl-v1.py`, `psl_inference.py`, `psl_predict.py` | Live webcam, MediaPipe hand landmarks, TFLite predict per frame, Urdu mapping via `PSL_TO_URDU`, edge-tts for speech, arabic-reshaper/bidi for rendering. Accumulates letters into words. |

Key properties of the current design that **won't carry over** to word-level:
- It classifies **one frame at a time** — no temporal context.
- Only **one hand, 42 features**. Word signs typically use **both hands and motion**.
- Normalization is a flat `StandardScaler` over the whole dataset — not robust to camera/subject scale.

Useful pieces we will reuse: MediaPipe capture loop, TFLite inference scaffolding, Urdu text rendering, TTS layer, label encoder pattern.

---

## 2. The Word-Level Data

Location: `data/archive/PakistanSignLanguageDataset/`

Two sibling directories, same schema, same class names:

| Dir | Words | Sequences / word | Frames / sequence | Feature dim |
|-----|-------|------------------|-------------------|-------------|
| `MP_Data/`        | ~64 (plus `stats.json`, `test_word`, `nothing`) | **50** | **60** | **126** |
| `MP_Data_mobile/` | 64                                              | **20** | **60** | **126** |

Each frame is saved as `<frame_idx>.npy`, shape `(126,)`, `float64`.

Decoding `126 = 2 × 21 × 3`: both hands (left + right) × 21 MediaPipe hand landmarks × (x, y, z). Observed:
- When only one hand is present, the other half of the vector is exactly `0.0` (padding, not NaN). Confirmed by inspecting a frame with exactly 63 non-zero values.
- Values are normalized image coords (`x, y ∈ [0, 1]`) plus a relative `z` (≈ depth from wrist). Range observed ≈ `[-0.07, 0.68]`.

Word vocabulary (from `MP_Data/` folder names):
> absolutely, aircrash, airplane, all, also, arrival, assalam-o-alaikum, atm, bald, beach, beak, bear, beard, bed, bench, bicycle, bird, both, bridge, bring, bulb, cartoon, chimpanzee, color_pencils, cow, crow, cupboard, deer, dog, donttouch, door, elephant, excuseme, facelotion, fan, garden, generator, goodbye, goodmorning, have_a_good_day, hello, ihaveacomplaint, left_hand, lifejacket, mine, mobile_phone, nailcutter, **nothing**, peacock, policecar, razor, s, shampoo, shower, sunglasses, **test_word**, thankyou, tissue, toothbrush, toothpaste, umbrella, water, we, welldone, you

Notes / gotchas:
- `stats.json` is **not** a class — filter it out.
- `nothing` is the standard "no-sign / idle" class — **keep** it, it is essential for continuous inference (see §4).
- `test_word` looks like a placeholder — drop it from training unless it has meaningful data.
- `MP_Data` (50 seqs, desktop) and `MP_Data_mobile` (20 seqs, mobile camera) share the same schema → **combine them**. Mobile sequences double as free domain-robustness augmentation.
- Expected totals per class: `50 + 20 = 70` sequences. Total training sequences ≈ `64 × 70 ≈ 4,500`. Modest but enough for a compact recurrent/transformer model with proper augmentation.

Sanity totals for training tensors:
- `X`: `(~4500, 60, 126)` float32
- `y`: `(~4500,)` int (up to 64 classes)

---

## 3. Recommended Model

Word-level sign recognition is a **sequence-classification** problem. Three reasonable choices, ranked for this dataset size:

### Primary recommendation: **1D-Conv frontend + BiLSTM + attention pooling**

Why it's the right first model here:
- Handles variable/short sequences well with very few parameters (~300–500 K).
- Conv1D smooths noisy per-frame landmarks and learns short motion primitives (hand open/close, path direction).
- BiLSTM captures the full temporal structure of a word; attention pooling gives a robust fixed-length embedding without averaging out the key motion moment.
- Trains in minutes on CPU, converts cleanly to TFLite for the existing inference scaffold.

Sketch:
```
Input (T=60, F=126)
  → Masking (to ignore zero-padded missing-hand frames if desired)
  → Conv1D(64, k=3, relu) → Conv1D(128, k=3, relu)
  → BiLSTM(128, return_sequences=True) → BiLSTM(64, return_sequences=True)
  → Attention pooling (softmax over time)
  → Dense(64, relu) → Dropout(0.3)
  → Dense(num_classes, softmax)
```

### Alternative: **Small Transformer encoder**
`d_model=128, heads=4, layers=3, ff=256`, sinusoidal positional encoding over T=60, CLS-token pooling. Slightly higher ceiling if we have time to tune, but needs more careful regularization at this data scale.

### Baseline to beat: **GRU-only** (`GRU(128) → GRU(64) → Dense`) — good smoke test, trains in under a minute.

### Data prep & augmentation (critical — matters more than model choice)
All applied on the `(T, 126)` tensor before training:
1. **Per-frame, per-hand normalization**: subtract the wrist landmark (index 0) from the other 20 of that hand, then scale by the max `|coord|` of that hand. Makes the model camera- and subject-agnostic. Done independently per hand; skip if the hand is all zeros.
2. **Drop the `z` channel** in a v1 model if signs are mostly planar — reduces noise. Keep `(T, 84)`. (Make it a config flag, try both.)
3. **Temporal jitter**: random uniform resample in `[0.8×, 1.2×]` length, then pad/truncate back to 60.
4. **Frame dropout**: randomly zero 5–10% of frames.
5. **Small Gaussian noise** on coords (σ ≈ 0.01).
6. **Horizontal mirror**: flip `x → 1 - x` and swap left/right hand halves. Only for signs that are genuinely handedness-agnostic — build a per-word allow-list (or skip this and rely on natural variance).
7. **Class balancing**: `class_weight='balanced'` in `.fit(...)`, since sequence counts may differ across words.

Train/val/test split: split **by sequence**, stratified by word, 70/15/15. Do **not** split by frame — that leaks.

Target metric: top-1 validation accuracy ≥ 85 % on 64 classes is a reasonable first bar; top-5 should be ≥ 95 %.

Export path: `.h5` + `.tflite` (same pattern as current `train_psl.py`), plus `label_encoder.pkl` and the per-hand normalization stats if any are global.

---

## 4. How Inference Should Work (Word-Level)

The existing alphabet loop (1 frame → 1 prediction) does **not** transfer. A word spans a full gesture in time, so inference is a **streaming sequence classification** problem.

### Runtime layout
1. **Capture**: MediaPipe Holistic (or two MediaPipe Hands instances) at ~20 FPS. Build the same 126-D vector per frame as in training. Missing hand → 63 zeros.
2. **Rolling buffer**: keep the last `W = 60` frames in a deque. On each new frame:
   - Apply the same per-hand normalization used in training.
   - Every `stride = 5` frames (≈ 4 predictions/sec), run the TFLite model on the current window.
3. **Idle gating with the `nothing` class**: the `nothing` class is the anchor that lets us tell "no sign happening" from a real word. Two options:
   - (a) If `argmax == nothing` or `max_prob < τ` (e.g., `τ = 0.7`), emit nothing.
   - (b) Use a lightweight motion detector (variance of wrist position over the window) as a pre-filter; only classify when the hands are actually moving. Cheaper than running the model every stride.
   Use both: motion gate first, then model, then threshold.
4. **Temporal smoothing**: maintain an EMA of the softmax vector across successive windows (`α = 0.6`), and only commit a word when:
   - same class wins `K = 3` consecutive windows, **and**
   - smoothed confidence ≥ `τ_commit` (e.g., 0.8), **and**
   - the word boundary is clear — i.e., the previous committed word was at least `cooldown = 1.0 s` ago, or a `nothing` window has occurred since.
5. **Emit**: append the committed word to a running sentence buffer. Map to Urdu via a new `PSL_WORD_TO_URDU` table (same shape as the existing `PSL_TO_URDU`). Render with arabic-reshaper + bidi. Trigger edge-tts (`ur-PK-UzmaNeural`) on the committed word — identical to the current alphabet TTS plumbing.
6. **UX polish**: reuse the existing overlay and TTS scaffolding from `src/inference/psl-v1.py`. Add a "current-window top-3" panel so debugging is easy. Add a "Clear sentence" / "Speak sentence" hotkey.

### Failure modes to watch for
- **Signer-handedness**: if the trained model saw only right-handed signers, a left-handed user will look mirrored. Mitigate with mirror augmentation for ambiguous signs, or detect dominant hand at startup and mirror the input if needed.
- **Partial occlusion**: MediaPipe drops frames where a hand leaves the frame — those become zeros. The Masking layer handles this during training; during inference, too many zero frames in a window (> 50 %) should suppress prediction.
- **Sign overlap**: back-to-back signs with no idle gap will blur. The `nothing` gate + cooldown addresses this; consider fine-tuning `stride` and `cooldown` from real-usage logs.

---

## 5. Concrete Next Steps (suggested order)

1. **New preprocessing script** `src/preprocessing/build_word_dataset.py`:
   - Walks `MP_Data/` and `MP_Data_mobile/`, skips `stats.json` and `test_word`.
   - Loads each sequence into a `(60, 126)` tensor, applies per-hand wrist-centered normalization, stacks into `X, y`.
   - Saves `data/psl_word_processed/{X.npy, y.npy, label_encoder.pkl, normalization.json}` with a 70/15/15 split by sequence.
2. **New training script** `src/training/train_psl_words.py`:
   - Builds the Conv1D+BiLSTM+attention model described above.
   - Uses `class_weight`, early stopping, `ReduceLROnPlateau`, batch size 32, ~80 epochs.
   - Saves `models/psl_words/{psl_word_classifier.h5, .tflite, training_curves.png, confusion_matrix.png}`.
3. **New inference script** `src/inference/psl_words_v1.py`:
   - Copy the UX skeleton of `psl-v1.py`.
   - Replace the per-frame predict with the 60-frame rolling-buffer logic from §4.
   - Add `PSL_WORD_TO_URDU` mapping.
4. **Leave the alphabet model alone.** Future work: a mode switch in the UI to toggle alphabet ↔ word mode, since they need different MediaPipe configs (Hands only vs Holistic / two-hand Hands) and different buffer logic.
5. **Evaluate on `MP_Data_mobile` only**, held out, to measure cross-device robustness.
