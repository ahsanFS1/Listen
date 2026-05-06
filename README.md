<div align="center">

# Listen
### Real-Time Pakistan Sign Language Recognition

*Bridging the communication gap through AI — one sign at a time.*

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![Flutter](https://img.shields.io/badge/Flutter-3.38%2B-02569B?style=flat-square&logo=flutter&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15--2.16-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-0097A7?style=flat-square&logo=google&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=flat-square&logo=fastapi&logoColor=white)
![Word Accuracy](https://img.shields.io/badge/Word%20Top--1-98.16%25-brightgreen?style=flat-square)
![Word Classes](https://img.shields.io/badge/Word%20Classes-64-blueviolet?style=flat-square)
![Alphabet Classes](https://img.shields.io/badge/Alphabet%20Classes-39-blueviolet?style=flat-square)

</div>

---

## What is Listen?

Listen is a real-time Pakistan Sign Language (PSL) recognition system. It watches a signer through a phone camera, recognises the signs they make, and translates them into written and spoken Urdu — live, with sub-second latency.

The project ships three coordinated components:

| Component | Purpose | Stack |
|-----------|---------|-------|
| **`server/`** | FastAPI WebSocket inference server (MediaPipe + TFLite) | Python 3.11, FastAPI, TensorFlow Lite, MediaPipe Hands |
| **`flutter_app/`** | Cross-platform mobile app (camera, UI, TTS, learn/quiz) | Flutter 3.38+, Dart 3.10+, Camera2, Supabase auth |
| **`src/`** | Reference desktop pipeline + training & preprocessing scripts | Python 3.10/3.11, TensorFlow, scikit-learn |

It currently supports two modes:

| Mode | Classes | Model | Test accuracy | Input |
|------|---------|-------|---------------|-------|
| **Words** | 64 PSL words | Conv1D + BiLSTM + Attention pooling | **98.16% top-1 · 99.39% top-5** | 60-frame × 126-D landmark sequence |
| **Alphabet** | 39 Urdu letters | MLP on per-frame landmarks | — | 42-D landmark vector (single hand, x/y) |

Words drawn from the official [PSL dictionary](https://psl.org.pk/dictionary).

---

## System architecture

```
┌──────────────────────┐    JPEG over WebSocket      ┌─────────────────────────┐
│  Flutter app         │  ─────────────────────────► │  FastAPI inference      │
│  (Android · iOS)     │      ws://host:8000          │  server                 │
│                      │      /ws/translate           │                         │
│  • Camera2 capture   │  ◄─────────────────────────  │  • MediaPipe Hands      │
│  • YUV→JPEG (Kotlin) │      JSON snapshot/frame    │    (model_complexity=0) │
│  • web_socket_channel│                              │  • TFLite interpreter   │
│  • flutter_tts (Urdu)│                              │  • FSM + EMA smoothing  │
│  • Supabase auth     │                              │  • TF Select ops (LSTM) │
│  • Neon suggestions  │  ──── REST /suggest/* ────►  │                         │
└──────────────────────┘                              └────────────┬────────────┘
                                                                   │
                                                                   ▼
                                                        ┌──────────────────────┐
                                                        │  Neon PostgreSQL     │
                                                        │  urdu_words /        │
                                                        │  urdu_sentences      │
                                                        └──────────────────────┘
```

The on-device pipeline was deliberately moved to the server: the MediaPipe Tasks API on Android uses a different hand model than `mp.solutions.hands(model_complexity=0)` and the classifier did not generalise. Running the exact training-time pipeline server-side gives the mobile app desktop-parity accuracy.

---

## Recognition pipeline (per frame)

```
Camera feed (Camera2, native YUV_420_888)
    │
    ▼
YUV → JPEG (Kotlin: YuvJpegPlugin.kt — quality 70, applies rotation)
    │
    ▼ binary WS frame
FastAPI WebSocket  /ws/translate?mode=words|alphabets
    │
    ▼
MediaPipe Hands  (model_complexity=0, max_num_hands=2, det/track=0.5)
    │
    ├─ Inverted handedness (training was on selfie-flipped frames):
    │      label "Right" ⇒ anatomical left-hand slot
    │
    ▼
Per-frame 126-D vector  =  lh(21·xyz) ⊕ rh(21·xyz)
    │
    ▼
Per-hand wrist-centred normalisation (translate wrist to origin, scale by max-abs)
    │
    ▼
Rolling 60-frame buffer  (~3 s @ 20 FPS)
    │
    ▼
TFLite invoke every 3 frames (TF Select ops for LSTM)
    │
    ▼
EMA-smoothed class probabilities  (α = 0.60)
    │
    ▼
Commit if top-1 ≥ 0.70 confidence and label ∉ {nothing, test_word, shower}
    │
    ▼
JSON snapshot → Flutter UI → flutter_tts (Urdu) + sentence strip
```

**Word-level commit rule.** A word commits when EMA-smoothed probability of the top class crosses `COMMIT_CONF_DEFAULT = 0.70`, after which the FSM enters an 0.8 s `COOLDOWN`. The motion gate (`MOTION_VAR_MIN = 1e-4`) suppresses inference on static frames.

**Alphabet commit rule.** A letter commits after `STABLE_REQUIRED = 40` consecutive frames at confidence ≥ 0.85, then a 0.4 s cooldown.

---

## Mobile app states

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
       │                                │ EMA conf ≥ 0.70
       │         hands leave            ▼
       └──────────────────── ┌──────────────────┐
                             │   COMMITTED      │
                             │  large glowing   │
                             │  word + Urdu TTS │
                             └──────────────────┘
```

After commit the user lowers their hands (natural word boundary). Raising hands again starts the next sign.

---

## Versions of core components

### Python server (`server/requirements.txt`)

| Package | Pinned spec | Notes |
|---------|-------------|-------|
| Python | **3.11** (3.10 also supported in `.python-version`) | TF + MediaPipe don't yet support 3.13+ |
| `tensorflow` | `>=2.15,<2.17` | TFLite interpreter; uses `SELECT_TF_OPS` for LSTM |
| `mediapipe` | `==0.10.14` | Hands solution, `model_complexity=0` (lite) |
| `opencv-python-headless` | `>=4.9` | JPEG decode + colour conversion |
| `numpy` | `>=1.26,<2.0` | Pinned below 2.0 for TF compatibility |
| `fastapi` | `>=0.110` | Async WebSocket + REST |
| `uvicorn[standard]` | `>=0.27` | ASGI server |
| `websockets` | `>=12` | Used by Uvicorn for WS |
| `scikit-learn` | `>=1.3` | `LabelEncoder` / `StandardScaler` (loaded via joblib) |
| `joblib` | `>=1.3` | Loads label encoder + scaler |
| `psycopg2-binary` | `>=2.9` | Neon Postgres for word/sentence suggestions |

### Training / preprocessing extras (`requirements-training.txt`, `requirements-preprocessing.txt`)

| Package | Why |
|---------|-----|
| `pandas`, `seaborn`, `matplotlib` | Training reports, confusion matrices |
| `tqdm` | Progress bars during landmark extraction |
| `tensorboard` | Optional training visualisation |
| `mediapipe`, `tensorflow` | Same versions as server |

### Reference desktop inference (`requirements-inference.txt`)

Adds: `arabic-reshaper`, `python-bidi` (Urdu rendering on OpenCV windows), `edge-tts`, `pygame` (Urdu TTS playback via `ur-PK-UzmaNeural`), `groq` (optional translation fallback), `gTTS`.

### Flutter app (`flutter_app/pubspec.yaml`, resolved by `pubspec.lock`)

| SDK / Package | Version | Purpose |
|---|---|---|
| Flutter SDK | **>= 3.38.4** | Cross-platform UI (lock pin) |
| Dart SDK | **>= 3.10.3 < 4.0.0** | App language |
| `camera` | 0.11.4 | Plugin façade |
| `camera_android` | 0.10.10+16 | Forced via `AndroidCamera()` (Camera2, not CameraX) |
| `camera_platform_interface` | 2.13.0 | |
| `web_socket_channel` | 3.0.3 | WebSocket client to inference server |
| `flutter_tts` | 4.2.5 | Urdu TTS on-device |
| `supabase_flutter` | 2.12.4 | Email/password auth |
| `video_player` | 2.11.1 | Bundled `.mp4` sign demonstrations |
| `shared_preferences` | 2.5.5 | Local progress / settings |
| `url_launcher` | 6.3.2 | Open external dictionary links |
| `http` | 1.6.0 | REST calls to `/suggest/*` |
| `cupertino_icons` | 1.0.9 | |
| `flutter_lints` (dev) | 4.0.0 | |

### Android toolchain

| | |
|---|---|
| Application ID | `com.listen.psl.flutter_app` |
| `minSdk` | **24** (required for Camera2 / MediaPipe Tasks) |
| `compileSdk` / `targetSdk` | inherits from Flutter SDK |
| `sourceCompatibility` / `targetCompatibility` | **Java 17** |
| `jvmTarget` | **17** |
| Gradle wrapper | **8.14** |
| Native plugin | `YuvJpegPlugin.kt` — Kotlin, platform graphics only |
| Permissions | `CAMERA`, `INTERNET`, `usesCleartextTraffic="true"` (dev `ws://`) |

### Models (committed under `models/`)

| Path | Format | Contents |
|------|--------|----------|
| `models/psl_words/psl_word_classifier.tflite` | TFLite (TF Select ops) | Conv1D+BiLSTM+Attention, 64 classes |
| `models/psl_words/psl_word_classifier.h5` | Keras H5 | Same model, training-time format |
| `models/psl_words/label_encoder.pkl` | joblib | sklearn `LabelEncoder` over class names |
| `models/psl_words/training_summary.json` | JSON | Test accuracy, epochs run, class list |
| `models/psl_words/{training_curves,confusion_matrix}.png` | PNG | Training visualisations |
| `models/psl/psl_landmark_classifier.tflite` | TFLite | Alphabet MLP (42-D input, 39 classes) |
| `models/psl/{label_encoder,scaler}.pkl` | joblib | Encoder + `StandardScaler` for alphabet |
| `flutter_app/assets/models/hand_landmarker.task` | MediaPipe Tasks bundle | Bundled but unused in current branch (kept for reference) |

### External services

| Service | Used for | Configured in |
|---------|----------|---------------|
| **Supabase** | Email/password auth | [flutter_app/lib/config/supabase_config.dart](flutter_app/lib/config/supabase_config.dart) |
| **Neon PostgreSQL** | `urdu_words`, `urdu_sentences` for completion suggestions | `server/suggestions.py` (env-overridable: `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `PGSSLMODE`) |

---

## Word-level model architecture

```
Input (60 frames × 126 features)
  │
  ├─ Conv1D(64,  k=3, relu, l2=1e-5)      ← smooths per-frame noise,
  ├─ Conv1D(128, k=3, relu, l2=1e-5)        learns local motion primitives
  ├─ Dropout(0.3)
  │
  ├─ Bidirectional(LSTM(128, return_sequences=True))   ← full temporal arc,
  ├─ Bidirectional(LSTM(64,  return_sequences=True))     both directions
  │
  ├─ AttentionPooling                       ← softmax-weighted pool
  │    v       = tanh(x · W + b)
  │    scores  = v · u
  │    weights = softmax(scores, axis=time)
  │    output  = Σ(weights × x)
  │
  ├─ Dense(64, relu) + Dropout(0.3)
  └─ Dense(64, softmax)

Optimiser: Adam (lr=1e-3) + ReduceLROnPlateau
Loss:      sparse_categorical_crossentropy
Callbacks: EarlyStopping(patience=15, monitor=val_accuracy)
Batch:     32        Max epochs: 80     Best epoch: 41
Export:    .h5 + .tflite (SELECT_TF_OPS for LSTM)
```

**Training data:**

| | |
|---|---|
| Dataset | `MP_Data` (50 seq/word) + `MP_Data_mobile` (20 seq/word) |
| Excluded classes | `test_word` (junk / unlabelled motion) |
| Split | 70 / 15 / 15 stratified per class |
| Total sequences | ~4,500 |
| Test loss | 0.1451 |
| Test accuracy | **98.16% top-1 · 99.39% top-5** |

---

## Supported words (64 classes)

<details>
<summary>Click to expand</summary>

`absolutely` `aircrash` `airplane` `all` `also` `arrival` `assalam-o-alaikum` `atm` `bald` `beach` `beak` `bear` `beard` `bed` `bench` `bicycle` `bird` `both` `bridge` `bring` `bulb` `cartoon` `chimpanzee` `color_pencils` `cow` `crow` `cupboard` `deer` `dog` `donttouch` `door` `elephant` `excuseme` `facelotion` `fan` `garden` `generator` `goodbye` `goodmorning` `have_a_good_day` `hello` `ihaveacomplaint` `left_hand` `lifejacket` `mine` `mobile_phone` `nailcutter` `nothing` `peacock` `policecar` `razor` `s` `shampoo` `shower` `sunglasses` `thankyou` `tissue` `toothbrush` `toothpaste` `umbrella` `water` `we` `welldone` `you`

</details>

## Supported alphabet letters (39 classes)

<details>
<summary>Click to expand</summary>

`Ain` `Alif` `Alifmad` `Aray` `Bay` `Byeh` `Chay` `Cyeh` `Daal` `Dal` `Dochahay` `Fay` `Gaaf` `Ghain` `Hamza` `Hay` `Jeem` `Kaf` `Khay` `Kiaf` `Lam` `Meem` `Nuun` `Nuungh` `Pay` `Ray` `Say` `Seen` `Sheen` `Suad` `Taay` `Tay` `Tuey` `Wao` `Zaal` `Zaey` `Zay` `Zuad` `Zuey`

</details>

---

## Repository layout

```
Listen/
├── server/                                    FastAPI WebSocket server
│   ├── app.py                                 /ws/translate, /healthz, /suggest/*
│   ├── sign_session.py                        Word pipeline (mirrors psl_words_v2.py)
│   ├── alphabet_session.py                    Alphabet pipeline (mirrors psl-v1.py)
│   ├── suggestions.py                         Neon Postgres lookups
│   └── requirements.txt
│
├── flutter_app/                               Flutter mobile app
│   ├── lib/
│   │   ├── main.dart                          Bootstrap (forces Camera2 on Android)
│   │   ├── app.dart                           AuthGate + MainShell (4-tab nav)
│   │   ├── ml/
│   │   │   ├── sign_client.dart               WebSocket client (drops > 2 in-flight)
│   │   │   ├── prediction.dart
│   │   │   └── yuv_jpeg.dart                  Dart bridge to native encoder
│   │   ├── screens/
│   │   │   ├── auth_screen.dart
│   │   │   ├── translate_screen.dart          Live recognition UI
│   │   │   ├── learn_screen.dart              Letter / word learning
│   │   │   ├── quiz_screen.dart
│   │   │   ├── dictionary_screen.dart
│   │   │   ├── word_video_screen.dart
│   │   │   ├── letter_video_screen.dart
│   │   │   ├── fullscreen_video_screen.dart
│   │   │   └── profile_screen.dart
│   │   ├── services/
│   │   │   ├── auth_service.dart              Supabase
│   │   │   ├── settings_service.dart          Server URL, threshold, voice
│   │   │   ├── suggestion_service.dart        REST → /suggest/*
│   │   │   ├── progress_service.dart
│   │   │   └── streak_service.dart
│   │   ├── data/{signs,alphabets}.dart        Word/letter metadata
│   │   ├── widgets/{state_pill,confidence_bar_widget}.dart
│   │   ├── theme/app_colors.dart
│   │   └── config/supabase_config.dart
│   ├── android/app/src/main/kotlin/com/listen/psl/flutter_app/
│   │   ├── MainActivity.kt
│   │   └── YuvJpegPlugin.kt                   Native YUV_420_888 → JPEG
│   ├── assets/
│   │   ├── videos/         64 word demos (.mp4)
│   │   ├── videos/letters/ 40 letter demos (.mp4)
│   │   ├── images/         App imagery + onboarding
│   │   └── models/         Reference TFLite + hand_landmarker.task
│   ├── pubspec.yaml
│   └── pubspec.lock
│
├── src/                                       Reference desktop pipeline
│   ├── inference/
│   │   ├── psl_words_v2.py                    Threaded UI (canonical reference)
│   │   ├── psl_words_v1.py                    Earlier word UI
│   │   ├── psl-v1.py                          Alphabet UI
│   │   ├── psl-v1-enhanced.py
│   │   └── asl-v1.py                          ASL experimental
│   ├── training/
│   │   ├── train_psl_words.py                 ← word-level training
│   │   ├── train_psl.py                       Alphabet training
│   │   └── train.py
│   └── preprocessing/
│       ├── build_word_dataset.py              ← run first for word-level
│       ├── extract_landmarks_psl.py
│       ├── preprocess_landmarks_psl.py
│       ├── extract_landmarks.py
│       └── preprocess_landmarks.py
│
├── models/
│   ├── psl_words/                             Word classifier + encoder + curves
│   ├── psl/                                   Alphabet classifier + encoder + scaler
│   └── train_models/                          Auxiliary training artefacts
│
├── scripts/                                   One-off data utilities
│   ├── augment_psl_dataset.py
│   ├── analyze_augmentation_pattern.py
│   ├── check_psl_labels.py
│   ├── convert_videos_to_images.py
│   ├── scrape_psl_alphabets.py
│   ├── scrape_psl_videos.py
│   └── verify_training_ready.py               ... and more
│
├── demo_images/                               Screenshots
├── requirements.txt                           Superset (rarely needed)
├── requirements-inference.txt                 Reference desktop inference
├── requirements-training.txt                  Training only
├── requirements-preprocessing.txt             Dataset construction
├── SETUP.md                                   Server + app run guide
└── README.md
```

---

## Getting started

### Prerequisites

- **Python 3.11** (3.10 works; TF + MediaPipe do not support 3.13+)
- **Flutter 3.38+** with Dart 3.10+
- **Android Studio** (Hedgehog or newer) + Android SDK; physical device or emulator on **API 24+**
- **JDK 17** for the Android build
- Phone and dev machine on the **same Wi-Fi**, or USB cable + `adb reverse`
- Webcam (for the reference Python apps in `src/inference/`)

### 1. Clone

```bash
git clone https://github.com/ahsanFS1/Listen.git
cd Listen
```

### 2. Run the inference server

```bash
cd server
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                      # listens on 0.0.0.0:8000
```

The server boots both the word and alphabet TFLite models eagerly. On startup it prints the LAN URL the phone should connect to:

```
ws://<your-LAN-IP>:8000/ws/translate
```

Health check:

```bash
curl http://127.0.0.1:8000/healthz   # → {"ok": true}
```

### 3. Run the Flutter app

```bash
cd flutter_app
flutter pub get
flutter devices                                         # confirm phone listed
flutter run --dart-define=PSL_WS_URL=ws://<LAN-IP>:8000/ws/translate
```

Defaults to `ws://10.0.2.2:8000/ws/translate` (Android emulator). For a physical phone use your Mac's LAN IP (`ipconfig getifaddr en0` on macOS) or `adb reverse tcp:8000 tcp:8000`.

The app's first launch shows the auth screen (Supabase email/password). After sign-in the four tabs are: **Translate · Learn · Dictionary · Profile**.

### 4. Reference desktop pipeline (optional)

For the original desktop OpenCV UI:

```bash
pip install -r requirements-inference.txt
python src/inference/psl_words_v2.py     # word-level (threaded UI)
python src/inference/psl-v1.py            # alphabet
```

---

## Training from scratch

```bash
pip install -r requirements-preprocessing.txt
pip install -r requirements-training.txt

# Step 1 — build the word dataset (~2-3 min, one-shot)
python src/preprocessing/build_word_dataset.py

# Step 2 — train (early stops around epoch 41, ~10 min on CPU)
python src/training/train_psl_words.py
```

Both steps consume the raw `.npy` sequences in `data/archive/PakistanSignLanguageDataset/{MP_Data, MP_Data_mobile}` — no video files needed. Outputs land in `models/psl_words/`.

---

## API reference (server)

### `GET /healthz`

```json
{ "ok": true }
```

### `WS /ws/translate?mode=words|alphabets`

- **Client → server (binary):** raw JPEG bytes for one camera frame.
- **Client → server (text):** `"ping"` → server replies `{"pong": true}`.
- **Server → client (text):** JSON snapshot per frame:

```json
{
  "state": "SIGNING",
  "label": "hello",
  "english": "hello",
  "urdu": "ہیلو",
  "confidence": 0.92,
  "committed": false,
  "hasHands": true,
  "bufferFill": 60,
  "bufferCapacity": 60,
  "mode": "words",
  "error": null
}
```

`committed` is `true` for exactly one frame when a word/letter has just crossed the threshold.

### `GET /suggest/words?prefix=<str>&limit=<int>`

Prefix-then-substring lookup against `urdu_words` (Neon).

```json
{ "prefix": "ہی", "suggestions": ["ہیلو", "ہیرا", "..."] }
```

### `GET /suggest/sentences?prefix=<str>&limit=<int>`

Prefix → contains-prefix-as-prefix → substring → fallback against `urdu_sentences`.

---

## Inference controls (desktop reference UI)

| Key | Action |
|-----|--------|
| Raise hands | Begin signing |
| Lower hands | Commit word boundary |
| `C` | Clear sentence |
| `S` | Speak full sentence (Urdu Edge TTS — `ur-PK-UzmaNeural`) |
| `Q` | Quit |

---

## Key design decisions

**Server-side inference, on-device capture.** The MediaPipe Tasks API on Android uses a different hand model than `mp.solutions.hands(model_complexity=0)`. Running the exact training-time pipeline server-side is what gives the mobile app its desktop-parity accuracy.

**Camera2 backend forced.** [main.dart](flutter_app/lib/main.dart#L16-L19) explicitly installs `AndroidCamera()` instead of the default CameraX, because CameraX on some Samsung devices applies double JPEG compression that destroys throughput.

**Native YUV → JPEG.** [YuvJpegPlugin.kt](flutter_app/android/app/src/main/kotlin/com/listen/psl/flutter_app/YuvJpegPlugin.kt) avoids ferrying raw YUV planes through Dart. JPEG quality 70 is a deliberate trade-off — lower it to 50 if you see lag.

**Frame drop policy.** [SignClient](flutter_app/lib/ml/sign_client.dart#L36-L39) caps `_maxInFlight = 2`. Backlog only adds latency, so excess frames are dropped — accuracy depends on the rolling 60-frame *window*, not on every frame reaching the server.

**Inverted handedness.** MediaPipe's selfie-trained model labels hands as if looking *at* the user, so `"Right"` ⇒ anatomical left. `HANDS_INVERT_HANDEDNESS = True` undoes this so the per-hand 63-D slots match training.

**Wrist-centred normalisation, not StandardScaler.** Each frame is independently translated so the wrist sits at the origin, then scaled by max-abs extent. This makes the input position- and scale-invariant on the fly — `StandardScaler` over the full dataset can't adapt to where in the frame the signer happens to be.

**Suppressed classes.** `"shower"` is dropped at inference because it overfits and dominates ambiguous frames; the FSM falls back to the next-best prediction. `"test_word"` is excluded from training entirely.

**EMA smoothing + hard threshold + cooldown.** Probabilities are smoothed with `α = 0.60` so a single noisy frame can't trigger a commit; commits require crossing 0.70, then sit out an 0.8 s cooldown so a held sign can't fire twice.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| App stuck on "Connecting to inference server…" | Open `http://<server-ip>:8000/healthz` in the phone's browser. If unreachable, check Wi-Fi / firewall. |
| `CLEARTEXT communication not permitted` | Confirm `usesCleartextTraffic="true"` in [AndroidManifest.xml](flutter_app/android/app/src/main/AndroidManifest.xml). |
| Server: `ModuleNotFoundError: sklearn` | `pip install scikit-learn` (also in `server/requirements.txt`). |
| Server: `Address already in use` | Another process on port 8000 — `lsof -i :8000` and kill it, or change `PORT` in [server/app.py](server/app.py#L23). |
| Buffer fills but never predicts | Move your hand more — the motion gate (`MOTION_VAR_MIN = 1e-4`) suppresses near-static frames. |
| Predictions feel laggy | Drop JPEG quality (e.g. 70 → 50) in the Flutter capture path, or lower the camera preset. |
| `[suggest] DB connect failed` | Suggestions are a soft dependency — the WebSocket pipeline keeps working. Set `DB_*` env vars or ignore. |

---

## Roadmap

- [x] Flutter mobile app (Android — iOS planned)
- [x] Server-side inference with desktop-parity accuracy
- [x] Supabase auth, learn/quiz mode, alphabet videos, dictionary
- [x] Urdu word + sentence suggestions backed by Neon Postgres
- [ ] iOS build + signing
- [ ] Expand vocabulary beyond 64 words
- [ ] Sentence-level language model for auto-correction
- [ ] Two-person simultaneous recognition
- [ ] Web demo

---

## Datasets

The word-level dataset is the [Dynamic Pakistan Sign Language Dataset](https://www.kaggle.com/datasets/mohib123456/dynamic-word-level-pakistan-sign-language-dataset?resource=download), recorded by our team using MediaPipe Holistic across desktop and mobile devices to maximise environmental diversity.

The alphabet dataset is **UAlpha40**, a 40-class Urdu alphabet image dataset for PSL (39 classes survive label-cleaning).

---

## License

MIT
