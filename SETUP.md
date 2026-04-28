# Listen PSL — Setup & Run Guide

## What's in this repo

- `src/inference/psl_words_v2.py` — original Python desktop app. Reference implementation.
- `models/psl_words/` — trained TFLite classifier (64 classes) + label encoder + MediaPipe hand landmarker.
- `server/` — FastAPI WebSocket inference server (new).
- `flutter_app/` — Android mobile app (new).

## Architecture

```
Phone camera ── JPEG (native encoder) ── WebSocket ──► FastAPI server
                                                           │
                                                  MediaPipe Hands (model_complexity=0)
                                                           │
                                              126-D feature vec → 60-frame buffer
                                                           │
                                                       TFLite LSTM
                                                           │
                                                    FSM + EMA smoothing
                                                           │
Phone UI  ◄──────────── JSON snapshot per frame ◄──────────┘
```

The server runs the **exact same** MediaPipe + TFLite pipeline as `psl_words_v2.py`, so mobile accuracy matches desktop.

## What changed (this branch)

- Dropped on-device MediaPipe Tasks + TFLite — the Tasks API uses a different hand model than `mp.solutions.hands(model_complexity=0)` and the classifier doesn't generalize to it.
- Added `server/` with FastAPI + WebSocket endpoint `/ws/translate`.
- Added `YuvJpegPlugin.kt` — native Kotlin YUV_420_888 → JPEG encoder (with rotation).
- Added `SignClient` (Dart) — WebSocket client whose API mirrors the old `SignPipeline` so the UI is unchanged.
- Manifest: added `INTERNET` permission and `usesCleartextTraffic="true"` for dev `ws://` URLs.

## Prerequisites

- macOS / Linux / Windows
- Python 3.11 (TF + MediaPipe don't support 3.13+ yet)
- Android Studio (Hedgehog or newer) with Android SDK + an emulator or physical device (API 24+)
- Flutter SDK 3.0+
- Phone and dev machine on the **same Wi-Fi**, OR USB cable + `adb reverse`

## 1. Run the server

```bash
cd server
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                      # listens on 0.0.0.0:8000
```

Quick health check from another terminal:

```bash
curl http://127.0.0.1:8000/healthz   # → {"ok":true}
```

You should see `[server] model loaded: 64 classes` in the log.

## 2. Find the WebSocket URL

Pick one based on how the phone reaches the server:

| Setup                            | URL                                              |
| -------------------------------- | ------------------------------------------------ |
| Android emulator on the dev Mac  | `ws://10.0.2.2:8000/ws/translate` (default)      |
| Physical phone on same Wi-Fi     | `ws://<dev-machine-LAN-IP>:8000/ws/translate`    |
| Physical phone via USB           | use `adb reverse tcp:8000 tcp:8000` then default |

To find your Mac's LAN IP: `ipconfig getifaddr en0`.

## 3. Run the app from Android Studio

1. Open Android Studio → **Open** → select `flutter_app/` (the folder, not the repo root).
2. First time: Android Studio will prompt to fetch the Flutter SDK / Dart plugin. Accept.
3. Open `flutter_app/pubspec.yaml` and click **Pub get** at the top (or run `flutter pub get` in the embedded terminal).
4. Plug in your phone (USB debugging on) **or** start an emulator from **Device Manager**.
5. Top toolbar: pick the device, then click ▶ **Run**.

If you need a non-default WebSocket URL, set it before running. Either:

**a)** Edit the constant in `flutter_app/lib/screens/translate_screen.dart`:

```dart
const String _kDefaultWsUrl = String.fromEnvironment(
  'PSL_WS_URL',
  defaultValue: 'ws://192.168.1.29:8000/ws/translate',  // your IP
);
```

**b)** Or pass it as a build flag — in **Run → Edit Configurations…** add to **Additional run args**:

```
--dart-define=PSL_WS_URL=ws://192.168.1.29:8000/ws/translate
```

## 4. Run the app from the command line (alternative)

```bash
cd flutter_app
flutter pub get
flutter devices                                         # confirm phone listed
flutter run --dart-define=PSL_WS_URL=ws://192.168.1.29:8000/ws/translate
```

## 5. Use it

1. Open the app — first launch shows the onboarding screen, tap **Get Started**.
2. Tab → **Translate** → tap **Start Camera**.
3. Server log should print `ws connected: …<phone-ip>…`.
4. Sign one word at a time. The buffer-fill bar climbs to 60/60, then predictions appear.
5. Once confidence ≥ 0.70 the word commits, gets spoken in Urdu, and is added to the sentence strip.

## Troubleshooting

| Symptom                                   | Fix                                                                       |
| ----------------------------------------- | ------------------------------------------------------------------------- |
| App stuck at "Connecting to inference server…" | Check phone reaches server: open `http://<ip>:8000/healthz` in phone browser. |
| `CLEARTEXT communication not permitted`   | Confirm `usesCleartextTraffic="true"` in `AndroidManifest.xml`.           |
| Server: `ModuleNotFoundError: sklearn`    | `pip install scikit-learn` (also in `requirements.txt`).                  |
| Server: `Address already in use`          | Another process on port 8000 — `lsof -i :8000` and kill, or change port.  |
| Buffer fills but never predicts           | Move your hand more — motion gate (`MOTION_VAR_MIN=1e-4`) suppresses static frames. |
| Predictions feel laggy                    | Drop JPEG quality in `lib/screens/translate_screen.dart` (`quality: 70` → `50`), or lower camera preset. |

## Repo layout (changed paths only)

```
server/
├── app.py                    FastAPI + /ws/translate
├── sign_session.py           MediaPipe + TFLite + FSM (mirrors psl_words_v2.py)
├── requirements.txt
└── README.md

flutter_app/
├── android/app/src/main/kotlin/com/listen/psl/flutter_app/
│   ├── MainActivity.kt
│   └── YuvJpegPlugin.kt      Native YUV→JPEG encoder
├── lib/ml/
│   ├── prediction.dart
│   ├── sign_client.dart      WebSocket client
│   └── yuv_jpeg.dart         Dart bridge to encoder
└── lib/screens/translate_screen.dart   wired to SignClient
```
