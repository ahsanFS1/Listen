"""
Train a word-level PSL sequence classifier.

Model: Conv1D frontend + BiLSTM stack + attention pooling.
Input:  (T=60, F=126) normalized landmark sequences from
        data/psl_word_processed/ (produced by build_word_dataset.py).
Output: models/psl_words/{psl_word_classifier.h5, .tflite,
                         training_curves.png, confusion_matrix.png}
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
DATA_DIR = "data/psl_word_processed"
MODEL_DIR = "models/psl_words"
os.makedirs(MODEL_DIR, exist_ok=True)

# Classes to exclude from training (junk / non-semantic labels).
# The preprocessed .npy files still contain them; we filter at load time
# so we don't need to rerun preprocessing.
EXCLUDE_CLASSES = {"test_word"}

BATCH_SIZE = 32
EPOCHS = 80
LEARNING_RATE = 1e-3
CONV1_FILTERS = 64
CONV2_FILTERS = 128
LSTM1_UNITS = 128
LSTM2_UNITS = 64
DENSE_UNITS = 64
DROPOUT = 0.3
L2 = 1e-5


# ---------------------------------------------------------------------
# Attention pooling layer
# ---------------------------------------------------------------------
class AttentionPooling(layers.Layer):
    """Softmax-weighted pooling over the time dimension.

    Given a (batch, T, D) sequence, learns a scalar score per timestep
    and returns a (batch, D) context vector. Replaces plain mean/last-
    step pooling with something that can emphasize the informative
    moment of the sign.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        d = input_shape[-1]
        self.W = self.add_weight(
            shape=(d, d), initializer="glorot_uniform", name="W"
        )
        self.b = self.add_weight(
            shape=(d,), initializer="zeros", name="b"
        )
        self.u = self.add_weight(
            shape=(d,), initializer="glorot_uniform", name="u"
        )
        super().build(input_shape)

    def call(self, x):
        v = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        scores = tf.tensordot(v, self.u, axes=1)          # (batch, T)
        weights = tf.nn.softmax(scores, axis=1)           # (batch, T)
        weights = tf.expand_dims(weights, -1)             # (batch, T, 1)
        return tf.reduce_sum(x * weights, axis=1)         # (batch, D)


def build_model(T, F, num_classes):
    reg = tf.keras.regularizers.l2(L2)
    inputs = layers.Input(shape=(T, F), name="landmarks")

    x = layers.Conv1D(CONV1_FILTERS, 3, padding="same",
                      activation="relu", kernel_regularizer=reg)(inputs)
    x = layers.Conv1D(CONV2_FILTERS, 3, padding="same",
                      activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(DROPOUT)(x)

    x = layers.Bidirectional(
        layers.LSTM(LSTM1_UNITS, return_sequences=True,
                    kernel_regularizer=reg)
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(LSTM2_UNITS, return_sequences=True,
                    kernel_regularizer=reg)
    )(x)

    x = AttentionPooling(name="attention_pool")(x)
    x = layers.Dense(DENSE_UNITS, activation="relu",
                     kernel_regularizer=reg)(x)
    x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs, name="psl_word_classifier")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("=" * 70)
    print("PSL Word-Level Training")
    print("=" * 70)

    # 1. Load data
    print("\nLoading preprocessed tensors...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    src_encoder = joblib.load(os.path.join(DATA_DIR, "label_encoder.pkl"))

    # Filter excluded classes and re-index labels to a dense 0..N-1 space.
    if EXCLUDE_CLASSES:
        src_classes = list(src_encoder.classes_)
        drop_ids = {i for i, c in enumerate(src_classes) if c in EXCLUDE_CLASSES}
        keep_classes = [c for c in src_classes if c not in EXCLUDE_CLASSES]

        def filter_split(X, y):
            mask = ~np.isin(y, list(drop_ids))
            return X[mask], y[mask]

        before = len(y_train) + len(y_val) + len(y_test)
        X_train, y_train = filter_split(X_train, y_train)
        X_val,   y_val   = filter_split(X_val,   y_val)
        X_test,  y_test  = filter_split(X_test,  y_test)
        after = len(y_train) + len(y_val) + len(y_test)
        print(f"  Excluded classes: {sorted(EXCLUDE_CLASSES)}")
        print(f"  Samples dropped:  {before - after:,} / {before:,}")

        label_encoder = LabelEncoder()
        label_encoder.fit(keep_classes)
        # Map old ids -> new ids via class-name lookup
        old_to_new = {
            src_classes.index(c): new_id for new_id, c in enumerate(label_encoder.classes_)
        }
        remap = np.vectorize(old_to_new.get)
        y_train = remap(y_train).astype(np.int64)
        y_val   = remap(y_val).astype(np.int64)
        y_test  = remap(y_test).astype(np.int64)
    else:
        label_encoder = src_encoder

    _, T, F = X_train.shape
    num_classes = len(label_encoder.classes_)
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}, y_test:  {y_test.shape}")
    print(f"  Classes: {num_classes}")

    # 2. Device info
    gpus = tf.config.list_physical_devices("GPU")
    print(f"\n  GPUs visible to TF: {[g.name for g in gpus] or 'none'}")

    # 3. Build + compile model
    print("\nBuilding model...")
    model = build_model(T, F, num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5")],
    )
    model.summary()

    # 4. Class weights
    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=y_train,
    )
    class_weight = {i: float(w) for i, w in enumerate(class_weights_arr)}

    # 5. Train
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)

    cbs = [
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=15,
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5,
            min_lr=1e-6, verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=cbs,
        verbose=1,
    )

    # 6. Evaluate
    print("\n" + "=" * 70)
    print("Evaluation")
    print("=" * 70)
    test_metrics = model.evaluate(X_test, y_test, verbose=0, return_dict=True)
    print(f"  test_loss:     {test_metrics['loss']:.4f}")
    print(f"  test_accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"  test_top5:     {test_metrics['top5']*100:.2f}%")

    y_pred_probs = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    y_pred = y_pred_probs.argmax(axis=1)

    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred,
        labels=np.arange(num_classes),
        target_names=label_encoder.classes_,
        zero_division=0,
    ))

    # 7. Plots
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Val")
    plt.title("Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.title("Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    curves_path = os.path.join(MODEL_DIR, "training_curves.png")
    plt.savefig(curves_path, dpi=150)
    plt.close()
    print(f"\nSaved training curves -> {curves_path}")

    plt.figure(figsize=(max(12, num_classes * 0.25),
                        max(10, num_classes * 0.25)))
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(num_classes))
    sns.heatmap(
        cm, annot=False, cmap="Blues", fmt="d",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
    )
    plt.title("PSL Word Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(rotation=90); plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix -> {cm_path}")

    # 8. Save models
    keras_path = os.path.join(MODEL_DIR, "psl_word_classifier.h5")
    tflite_path = os.path.join(MODEL_DIR, "psl_word_classifier.tflite")
    model.save(keras_path)
    print(f"Saved Keras model     -> {keras_path}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,  # needed for LSTM
    ]
    tflite_bytes = converter.convert()
    with open(tflite_path, "wb") as fh:
        fh.write(tflite_bytes)
    print(f"Saved TFLite model    -> {tflite_path}")

    # Also copy label encoder alongside the model for inference convenience
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

    # Persist a small run summary
    summary = {
        "test_loss": float(test_metrics["loss"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_top5": float(test_metrics["top5"]),
        "epochs_run": len(history.history["loss"]),
        "num_classes": int(num_classes),
        "classes": list(label_encoder.classes_),
    }
    with open(os.path.join(MODEL_DIR, "training_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 70)
    print("Training complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
