<div align="center">

# Listen
### Real-Time Pakistan Sign Language Recognition

*Bridging the communication gap through AI — one sign at a time.*

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-0097A7?style=flat-square&logo=google&logoColor=white)
![Accuracy](https://img.shields.io/badge/Word%20Accuracy-98.16%25-brightgreen?style=flat-square)
![Classes](https://img.shields.io/badge/Word%20Classes-64-blueviolet?style=flat-square)

</div>

---

## What is Listen?

Listen is a real-time Pakistan Sign Language (PSL) recognition system. It watches your hands through a camera, recognises the signs you make, and translates them into written and spoken Urdu — live, on device, with no internet required.

It currently supports two modes:

| Mode | Classes | Model | Accuracy |
|------|---------|-------|----------|
| **Alphabet** | 40 Urdu letters | MLP on per-frame landmarks | — |
| **Word** | 64 PSL words | Conv1D + BiLSTM + Attention | **98.16% top-1** |

The word-level model is the main focus of active development. The full list of supported words spans greetings, animals, everyday objects, transport, and common phrases — all drawn from the official [PSL dictionary](https://psl.org.pk/dictionary).

---

## How it works

```
Camera feed
    │
    ▼
MediaPipe Holistic  ──────────────────────────────────────────┐
(anatomical left/right hand landmark detection)               │
    │                                                         │
    ▼                                                         │
Per-frame feature vector  (126-D)                             │
  └── Left hand:  21 landmarks × (x, y, z)  =  63 values     │
  └── Right hand: 21 landmarks × (x, y, z)  =  63 values     │
    │                                                         │
    ▼                                                         │
Per-hand wrist-centred normalisation                          │
  └── Translate wrist to origin, scale by max-abs extent      │
  └── Missing hand → leave as zeros (preserves signal)        │
    │                                                         │
    ▼                                                         │
Rolling 60-frame buffer  (≈ 3 s at 20 FPS)                   │
    │                                                         │
    ▼                                                         │
TFLite model  (every 5 frames)                               │
  Conv1D(64) → Conv1D(128)                                    │
  BiLSTM(128) → BiLSTM(64)                                    │
  Attention Pooling → Dense(64) → Softmax(64)                 │
    │                                                         │
    ▼                                                         │
EMA-smoothed prediction                                       │
  └── K=3 consecutive high-confidence windows → commit        │
  └── nothing class + motion gate → idle detection            │
    │                                                         │
    ▼                                                         │
Committed word  ──►  Urdu text  ──►  Edge TTS (ur-PK-UzmaNeural)
```

---

## Inference UI

The real-time interface runs a four-state machine designed around the natural rhythm of signing:

```
  ┌──────────┐    hands appear    ┌─────────────┐
  │   IDLE   │ ─────────────────► │  BUFFERING  │
  │  pulsing │                    │  arc fills  │
  │   ring   │ ◄──────────────    │  60 frames  │
  └──────────┘  hands leave       └──────┬──────┘
       ▲                                 │ buffer full
       │                                 ▼
       │ hands leave           ┌─────────────────┐
       │                       │   PREDICTING    │
       │                       │  glowing border │
       │                       │  live top-1 +   │
       │                       │  confidence bar │
       │                       └────────┬────────┘
       │                                │ K consecutive
       │                                │ high-conf windows
       │         hands leave            ▼
       └──────────────────── ┌──────────────────┐
                             │   COMMITTED      │
                             │  large glowing   │
                             │  word + Urdu TTS │
                             └──────────────────┘
```

After a word is committed, the user lowers their hands — this acts as a natural word boundary. Raising hands again starts the next sign.

---

## Supported words (64 classes)

<details>
<summary>Click to expand</summary>

`absolutely` `aircrash` `airplane` `all` `also` `arrival` `assalam-o-alaikum` `atm` `bald` `beach` `beak` `bear` `beard` `bed` `bench` `bicycle` `bird` `both` `bridge` `bring` `bulb` `cartoon` `chimpanzee` `color_pencils` `cow` `crow` `cupboard` `deer` `dog` `donttouch` `door` `elephant` `excuseme` `facelotion` `fan` `garden` `generator` `goodbye` `goodmorning` `have_a_good_day` `hello` `ihaveacomplaint` `left_hand` `lifejacket` `mine` `mobile_phone` `nailcutter` `nothing` `peacock` `policecar` `razor` `s` `shampoo` `shower` `sunglasses` `thankyou` `tissue` `toothbrush` `toothpaste` `umbrella` `water` `we` `welldone` `you`

</details>

---

## Model architecture

```
Input (60 frames × 126 features)
  │
  ├─ Conv1D(64 filters, k=3, relu)       ← smooths per-frame noise,
  ├─ Conv1D(128 filters, k=3, relu)         learns local motion primitives
  ├─ Dropout(0.3)
  │
  ├─ BiLSTM(128 units, return_sequences) ← captures full temporal arc
  ├─ BiLSTM(64 units, return_sequences)     of a sign in both directions
  │
  ├─ Attention Pooling                   ← weights the most informative
  │    score = softmax(tanh(x·W + b)·u)     moment in the sequence
  │    output = Σ(score × x)
  │
  ├─ Dense(64, relu) + Dropout(0.3)
  └─ Dense(64, softmax)

Parameters: ~380 K
```

**Training details:**

| | |
|---|---|
| Dataset | MP_Data (50 seq/word) + MP_Data_mobile (20 seq/word) |
| Split | 70 / 15 / 15 by sequence, stratified |
| Total sequences | ~4,500 |
| Optimiser | Adam (lr=1e-3, ReduceLROnPlateau) |
| Early stopping | patience=15 on val_accuracy |
| Best epoch | 26 / 80 |
| Test accuracy | **98.16% top-1 · 99.39% top-5** |
| Export | `.h5` + `.tflite` (SELECT_TF_OPS for LSTM) |

---

## Project structure

```
Listen/
├── data/
│   ├── archive/
│   │   └── PakistanSignLanguageDataset/
│   │       ├── MP_Data/           # 65 word classes × 50 sequences × 60 frames
│   │       └── MP_Data_mobile/    # 63 word classes × 20 sequences × 60 frames
│   ├── psl_word_processed/        # preprocessed .npy tensors (after build_word_dataset.py)
│   └── UAlpha40 .../              # 40-class Urdu alphabet image dataset
│
├── models/
│   ├── psl/                       # alphabet classifier (.h5, .tflite)
│   └── psl_words/                 # word classifier (.h5, .tflite, label_encoder.pkl)
│
├── src/
│   ├── preprocessing/
│   │   ├── build_word_dataset.py        # ← run first for word-level
│   │   ├── extract_landmarks_psl.py     # alphabet landmark extraction
│   │   └── preprocess_landmarks_psl.py  # alphabet preprocessing
│   ├── training/
│   │   ├── train_psl_words.py     # ← word-level training
│   │   └── train_psl.py           # alphabet training
│   └── inference/
│       ├── psl_words_v1.py        # ← word-level live inference (main)
│       └── psl-v1.py              # alphabet live inference
│
└── plan.md                        # architecture decisions & design notes
```

---

## Getting started

### Prerequisites

- Python 3.10+
- Webcam (or phone camera via OBS/DroidCam)
- macOS / Linux / Windows

### 1. Clone

```bash
git clone https://github.com/ahsanFS1/Listen.git
cd Listen
```

### 2. Install dependencies

```bash
# For inference only (recommended if you just want to run it)
pip install -r requirements-inference.txt

# For training
pip install -r requirements-training.txt

# For preprocessing
pip install -r requirements-preprocessing.txt
```

### 3. Run inference (pre-trained model included)

```bash
python src/inference/psl_words_v1.py
```

The model files in `models/psl_words/` are committed to the repo — you can run inference immediately without training.

---

## Training from scratch

```bash
# Step 1 — build the word dataset (takes ~2-3 min, runs once)
python src/preprocessing/build_word_dataset.py

# Step 2 — train (early stops around epoch 26, ~5-10 min on CPU)
python src/training/train_psl_words.py
```

Both steps use the raw `.npy` sequences in `data/archive/` — no video files needed.

---

## Inference controls

| Key | Action |
|-----|--------|
| Raise hands | Begin signing |
| Lower hands | Commit word boundary |
| `C` | Clear sentence |
| `S` | Speak full sentence (Urdu TTS) |
| `Q` | Quit |

---

## Key design decisions

**Why MediaPipe Holistic instead of Hands?**
The training data was extracted using Holistic's `left_hand_landmarks` / `right_hand_landmarks` — anatomically assigned slots. Using `mp.solutions.hands` at inference inverts the slot assignment for some signs, causing a feature mismatch that hurts accuracy for two-handed and ambiguous-orientation signs. Holistic eliminates this.

**Why process the un-flipped frame?**
MediaPipe's handedness classifier assumes a non-mirrored image. Passing a flipped webcam frame scrambles Left/Right labels. Landmarks are extracted from the raw frame; the display is mirrored separately for the user's comfort.

**Why drop `test_word`?**
Its training data is a mix of empty frames and unlabelled random motion — not a coherent class. At inference it became a catch-all for anything ambiguous. Removing it forces confident predictions into real word classes, with `nothing` handling genuine idle frames.

**Why wrist-centred normalisation instead of StandardScaler?**
StandardScaler normalises across the whole dataset — it can't adapt to where in the frame your hands happen to be during a live session. Wrist-centred normalisation makes each frame position- and scale-invariant on the fly, matching the training normalisation exactly.

---

## Roadmap

- [ ] Flutter mobile app (iOS + Android) — model already exported as `.tflite`
- [ ] Expand vocabulary beyond 64 words
- [ ] Sentence-level language model for auto-correction
- [ ] Two-person simultaneous recognition
- [ ] Web demo

---

## Dataset

The word-level dataset is the [Dynamic Pakistan Sign Language Dataset](https://www.kaggle.com/datasets/mohib123456/dynamic-word-level-pakistan-sign-language-dataset?resource=download), recorded by our team using MediaPipe Holistic across desktop and mobile devices to maximise environmental diversity.

The alphabet dataset is [UAlpha40](https://github.com/), a comprehensive Urdu alphabet image dataset for PSL.

---

## License

MIT

