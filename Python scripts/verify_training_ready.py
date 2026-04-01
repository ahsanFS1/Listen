"""
Pre-training verification script for PSL model.
Checks all requirements before starting training.
"""

import os
import pandas as pd
import joblib
import tensorflow as tf

print("=" * 70)
print("PSL Training Pre-Flight Check")
print("=" * 70)

errors = []
warnings = []

# ---------------------------------------------------------------------
# 1. Check preprocessed data files
# ---------------------------------------------------------------------
print("\nChecking preprocessed data files...")
DATA_DIR = "data/psl_processed"
required_files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv", 
                  "label_encoder.pkl", "scaler.pkl"]

for filename in required_files:
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
            print(f"   [OK] {filename:20s} - Shape: {df.shape}")
        else:
            print(f"   [OK] {filename:20s} - Exists")
    else:
        errors.append(f"Missing file: {filepath}")
        print(f"   [FAIL] {filename:20s} - NOT FOUND")

# ---------------------------------------------------------------------
# 2. Check data dimensions match
# ---------------------------------------------------------------------
print("\nVerifying data consistency...")
try:
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))
    
    if X_train.shape[0] == y_train.shape[0]:
        print(f"   [OK] Training data: {X_train.shape[0]:,} samples match")
    else:
        errors.append(f"Training data mismatch: X={X_train.shape[0]}, y={y_train.shape[0]}")
        
    if X_test.shape[0] == y_test.shape[0]:
        print(f"   [OK] Test data: {X_test.shape[0]:,} samples match")
    else:
        errors.append(f"Test data mismatch: X={X_test.shape[0]}, y={y_test.shape[0]}")
        
    if X_train.shape[1] == 42:
        print(f"   [OK] Feature dimensions: {X_train.shape[1]} (21 landmarks × 2)")
    else:
        errors.append(f"Wrong feature dimensions: {X_train.shape[1]} (expected 42)")
        
except Exception as e:
    errors.append(f"Error loading data: {e}")

# ---------------------------------------------------------------------
# 3. Check label encoder
# ---------------------------------------------------------------------
print("\nChecking label encoder...")
try:
    label_encoder = joblib.load(os.path.join(DATA_DIR, "label_encoder.pkl"))
    num_classes = len(label_encoder.classes_)
    print(f"   [OK] Number of classes: {num_classes}")
    print(f"   [OK] Classes: {', '.join(sorted(label_encoder.classes_)[:5])}...")
    
    if num_classes < 2:
        errors.append(f"Too few classes: {num_classes}")
    elif num_classes > 50:
        warnings.append(f"Unusually high number of classes: {num_classes}")
        
except Exception as e:
    errors.append(f"Error loading label encoder: {e}")

# ---------------------------------------------------------------------
# 4. Check model save directory
# ---------------------------------------------------------------------
print("\nChecking model save paths...")
MODEL_DIR = "models/psl"
try:
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"   [OK] Model directory: {MODEL_DIR}")
    
    # Check if we can write to the directory
    test_file = os.path.join(MODEL_DIR, ".test_write")
    with open(test_file, 'w') as f:
        f.write("test")
    os.remove(test_file)
    print(f"   [OK] Write permissions: OK")
    
    # Check model save paths
    keras_path = os.path.join(MODEL_DIR, "psl_landmark_classifier.h5")
    tflite_path = os.path.join(MODEL_DIR, "psl_landmark_classifier.tflite")
    print(f"   [OK] Keras model path: {keras_path}")
    print(f"   [OK] TFLite model path: {tflite_path}")
    
except Exception as e:
    errors.append(f"Error with model directory: {e}")

# ---------------------------------------------------------------------
# 5. Check TensorFlow and GPU
# ---------------------------------------------------------------------
print("\nChecking TensorFlow and hardware...")
print(f"   [OK] TensorFlow version: {tf.__version__}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"   [OK] GPU available: {gpus[0].name}")
else:
    warnings.append("No GPU detected - training will use CPU (slower)")
    print(f"   [WARN] No GPU detected - will use CPU")

# ---------------------------------------------------------------------
# 6. Test model creation
# ---------------------------------------------------------------------
print("\nTesting model creation...")
try:
    test_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(42,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    test_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    print(f"   [OK] Model architecture: OK")
    print(f"   [OK] Model compilation: OK")
    print(f"   [OK] Total parameters: {test_model.count_params():,}")
    
except Exception as e:
    errors.append(f"Error creating model: {e}")

# ---------------------------------------------------------------------
# 7. Test TFLite conversion
# ---------------------------------------------------------------------
print("\nTesting TFLite conversion...")
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(test_model)
    tflite_model = converter.convert()
    print(f"   [OK] TFLite conversion: OK")
    print(f"   [OK] TFLite model size: {len(tflite_model):,} bytes")
except Exception as e:
    errors.append(f"TFLite conversion failed: {e}")

# ---------------------------------------------------------------------
# Final Report
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("PRE-FLIGHT CHECK SUMMARY")
print("=" * 70)

if errors:
    print("\n[FAIL] ERRORS FOUND:")
    for i, error in enumerate(errors, 1):
        print(f"   {i}. {error}")
    print("\n[WARN] CANNOT START TRAINING - Fix errors first!")
else:
    print("\n[OK] All critical checks passed!")
    
if warnings:
    print("\n[WARN] WARNINGS:")
    for i, warning in enumerate(warnings, 1):
        print(f"   {i}. {warning}")

if not errors:
    print("\n" + "=" * 70)
    print("READY TO START TRAINING!")
    print("=" * 70)
    print("\nRun: python src/training/train_psl.py")
    print("\nExpected training time:")
    if gpus:
        print("   - With GPU: ~5-15 minutes (50 epochs)")
    else:
        print("   - With CPU: ~30-60 minutes (50 epochs)")
    print("\n" + "=" * 70)

