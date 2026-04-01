# Listen — Pakistan Sign Language (PSL) Recognition

A real-time sign language recognition system that detects **40 Urdu alphabet hand signs** via webcam, converts them to Urdu script, and speaks them aloud using text-to-speech.

---

## How It Works

```
Raw Dataset → Extract Landmarks → Preprocess → Train Model → Run Inference
```

- **MediaPipe** detects hand landmarks from webcam/images
- A **TensorFlow Lite** neural network classifies 40 Urdu alphabet classes
- Recognized letters are assembled into words, displayed in Urdu script, and spoken via **Edge TTS**

---

## Requirements

- Python 3.11
- Webcam

Install dependencies:
```bash
pip install -r requirements-inference.txt
```

---

## Running Inference (Main App)

> ⚠️ You need the trained model files first (see below).

```bash
python src/inference/psl-v1.py
```

**Controls:**
| Key | Action |
|-----|--------|
| `q` | Quit |
| `c` | Clear sentence |
| `Space` | Add word break + speak word |
| `1`–`5` | Select word prediction |

---

## Model Files (Required)

The trained model is **not included in the repository**. You need these 3 files:

| File | Path |
|------|------|
| TFLite model | `models/psl/psl_landmark_classifier.tflite` |
| Label encoder | `src/data/psl_processed/label_encoder.pkl` |
| Scaler | `src/data/psl_processed/scaler.pkl` |

Get them from a teammate who has already trained the model, or re-train from scratch (see below).

---

## Re-Training from Scratch

Only needed if you have the **UAlpha40 dataset**. Place it at:
```
data/UAlpha40 A Comprehensive Dataset of Urdu alphabets for Pakistan Sign Language/
```

Then run the pipeline in order:

```bash
# Step 1 — Extract hand landmarks from dataset images
pip install -r requirements-preprocessing.txt
python src/preprocessing/extract_landmarks_psl.py

# Step 2 — Preprocess & split data
python src/preprocessing/preprocess_landmarks_psl.py

# Step 3 — Train the model (~50 epochs)
pip install -r requirements-training.txt
python src/training/train_psl.py
```

The trained model will be saved to `models/psl/`.

---

## Project Structure

```
Listen/
├── src/
│   ├── inference/         # Run the live recognition app
│   ├── preprocessing/     # Extract & preprocess landmarks
│   └── training/          # Train the TFLite model
├── models/psl/            # Trained model files (not in git)
├── Python scripts/        # Utility/dataset management scripts
└── requirements-*.txt     # Dependencies per stage
```
