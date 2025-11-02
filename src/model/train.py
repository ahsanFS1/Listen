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
DATA_DIR = "data/processed"
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# 📂 Load preprocessed data
# ---------------------------------------------------------------------
print("Loading data...")
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv")).values
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv")).values
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

label_encoder = joblib.load(os.path.join(DATA_DIR, "label_encoder.pkl"))
num_classes = len(label_encoder.classes_)
print(f"✅ Loaded {X_train.shape[0]} training and {X_test.shape[0]} testing samples.")
print("Classes:", list(label_encoder.classes_))

# ---------------------------------------------------------------------
# 🧱 Define the model
# ---------------------------------------------------------------------

physical_devices = tf.config.list_physical_devices()
print("\n🔍 Available devices:")
for device in physical_devices:
    print(f"• {device.device_type}: {device.name}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"\n🚀 Using GPU: {gpus[0].name}\n")
else:
    print("\n⚠️ No GPU detected — using CPU instead.\n")

    
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(42,)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# ---------------------------------------------------------------------
# 🚀 Train the model
# ---------------------------------------------------------------------
print("\nTraining model...\n")
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=40,
    batch_size=64,
    verbose=1
)

# ---------------------------------------------------------------------
# 📊 Evaluate performance
# ---------------------------------------------------------------------
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test accuracy: {test_acc*100:.2f}%")

# Plot accuracy/loss curves
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "training_curves.png"))
plt.close()

# ---------------------------------------------------------------------
# 🧮 Classification report
# ---------------------------------------------------------------------
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))

# Confusion Matrix
plt.figure(figsize=(12,10))
sns.heatmap(confusion_matrix(y_test, y_pred_classes), annot=False, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))
plt.close()

# ---------------------------------------------------------------------
# 💾 Save model + TFLite
# ---------------------------------------------------------------------
keras_path = os.path.join(MODEL_DIR, "asl_landmark_classifier.h5")
tflite_path = os.path.join(MODEL_DIR, "asl_landmark_classifier.tflite")

model.save(keras_path)
print(f"✅ Saved Keras model → {keras_path}")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(tflite_path, "wb").write(tflite_model)
print(f"✅ Saved TFLite model → {tflite_path}")
