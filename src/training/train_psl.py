"""
Train a neural network classifier for PSL (Pakistan Sign Language) recognition.
Uses preprocessed landmark data to train a model for 40 Urdu alphabet classes.
"""

import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ---------------------------------------------------------------------
# ⚙️ Configuration
# ---------------------------------------------------------------------
DATA_DIR = "data/psl_processed"
MODEL_DIR = "models/psl"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# 📂 Load preprocessed data
# ---------------------------------------------------------------------
print("=" * 70)
print("🚀 PSL Model Training")
print("=" * 70)
print(f"\nData directory: {DATA_DIR}")
print(f"Model directory: {MODEL_DIR}\n")
print("=" * 70 + "\n")

print("📂 Loading preprocessed data...")
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv")).values
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv")).values
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

label_encoder = joblib.load(os.path.join(DATA_DIR, "label_encoder.pkl"))
num_classes = len(label_encoder.classes_)

print(f"✅ Loaded {X_train.shape[0]:,} training samples")
print(f"✅ Loaded {X_test.shape[0]:,} testing samples")
print(f"\n📋 Training for {num_classes} Urdu alphabet classes:")
print(f"   {', '.join(sorted(label_encoder.classes_)[:10])}...")

# ---------------------------------------------------------------------
# 🧱 Check for GPU
# ---------------------------------------------------------------------
physical_devices = tf.config.list_physical_devices()
print("\n🔍 Available devices:")
for device in physical_devices:
    print(f"   • {device.device_type}: {device.name}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"\n🚀 Using GPU: {gpus[0].name}\n")
else:
    print("\n⚠️  No GPU detected — using CPU instead.\n")

# ---------------------------------------------------------------------
# 🧱 Define the model
# ---------------------------------------------------------------------
print("🧱 Building neural network model...")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(42,)),  # 21 landmarks × 2 coordinates
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n📊 Model Summary:")
model.summary()

# ---------------------------------------------------------------------
# 🚀 Train the model
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("🚀 Starting Training")
print("=" * 70 + "\n")

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=64,
    verbose=1
)

# ---------------------------------------------------------------------
# 📊 Evaluate performance
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("📊 Evaluating Model")
print("=" * 70 + "\n")

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Test Loss: {test_loss:.4f}")
print(f"✅ Test Accuracy: {test_acc*100:.2f}%")

# Plot accuracy/loss curves
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "training_curves.png"), dpi=150)
print(f"\n✅ Saved training curves to {MODEL_DIR}/training_curves.png")

# ---------------------------------------------------------------------
# 🧮 Classification report
# ---------------------------------------------------------------------
print("\n📋 Generating classification report...")
y_pred = model.predict(X_test, verbose=0)
y_pred_classes = y_pred.argmax(axis=1)

print("\n" + "=" * 70)
print("Classification Report:")
print("=" * 70)
print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))

# Confusion Matrix
print("\n📊 Generating confusion matrix...")
plt.figure(figsize=(16, 14))
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=False, cmap="Blues", fmt="d",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("PSL Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)
print(f"✅ Saved confusion matrix to {MODEL_DIR}/confusion_matrix.png")

# ---------------------------------------------------------------------
# 💾 Save model + TFLite
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("💾 Saving Models")
print("=" * 70 + "\n")

keras_path = os.path.join(MODEL_DIR, "psl_landmark_classifier.h5")
tflite_path = os.path.join(MODEL_DIR, "psl_landmark_classifier.tflite")

model.save(keras_path)
print(f"✅ Saved Keras model → {keras_path}")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(tflite_path, "wb").write(tflite_model)
print(f"✅ Saved TFLite model → {tflite_path}")

print("\n" + "=" * 70)
print("🎉 Training Complete!")
print("=" * 70)
print(f"\n📈 Final Results:")
print(f"   • Test Accuracy: {test_acc*100:.2f}%")
print(f"   • Classes: {num_classes} Urdu alphabets")
print(f"   • Model: {keras_path}")
print(f"   • TFLite: {tflite_path}")
print("\n" + "=" * 70)

