"""
Build a word-level PSL dataset from the MP_Data archive.

Walks data/archive/PakistanSignLanguageDataset/{MP_Data, MP_Data_mobile},
loads each sequence as a (T=60, F=126) tensor, applies per-hand
wrist-centered normalization, and writes stratified 70/15/15 splits.

Layout of the 126-D per-frame vector (MediaPipe Holistic convention):
    [lh_x0, lh_y0, lh_z0, ..., lh_x20, lh_y20, lh_z20,
     rh_x0, rh_y0, rh_z0, ..., rh_x20, rh_y20, rh_z20]
When a hand is not detected in a frame, its 63 values are all zero.
"""

import os
import json
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
DATA_ROOTS = [
    "data/archive/PakistanSignLanguageDataset/MP_Data",
    "data/archive/PakistanSignLanguageDataset/MP_Data_mobile",
]
OUTPUT_DIR = "data/psl_word_processed"
T = 60          # frames per sequence
F = 126         # features per frame (2 hands * 21 landmarks * xyz)
N_HANDS = 2
N_LANDMARKS = 21
N_COORDS = 3

# Splits (applied per-class so every word is present in each split)
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
RANDOM_STATE = 42

# Directories that are not real classes
SKIP_ENTRIES = {"stats.json"}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def list_sequence_dirs(word_dir):
    """Return sorted list of absolute paths to sequence subdirs inside a word dir."""
    seqs = []
    for name in os.listdir(word_dir):
        p = os.path.join(word_dir, name)
        if os.path.isdir(p):
            seqs.append(p)
    return sorted(seqs)


def load_sequence(seq_dir):
    """Load a sequence directory into a (T, 126) float32 tensor.

    Missing frame files and wrong-shape arrays are left as zeros.
    """
    out = np.zeros((T, F), dtype=np.float32)
    for t in range(T):
        p = os.path.join(seq_dir, f"{t}.npy")
        if not os.path.exists(p):
            continue
        arr = np.load(p)
        if arr.shape != (F,):
            continue
        out[t] = arr.astype(np.float32)
    return out


def normalize_frame(frame):
    """Per-hand wrist-centered, max-abs scaled normalization.

    Missing hands (all-zero halves) are preserved as zeros so the model
    keeps the missingness signal. A present hand is translated so its
    wrist sits at the origin, then scaled by the max absolute coordinate
    of that hand — making the representation invariant to camera
    position, signer distance, and subject scale.
    """
    f = frame.reshape(N_HANDS, N_LANDMARKS, N_COORDS).copy()
    for h in range(N_HANDS):
        hand = f[h]
        if not np.any(hand):
            continue
        wrist = hand[0].copy()
        hand -= wrist
        max_abs = np.max(np.abs(hand))
        if max_abs > 0:
            hand /= max_abs
        f[h] = hand
    return f.reshape(F)


def normalize_sequence(seq):
    """Apply normalize_frame to every frame of a (T, F) tensor."""
    out = np.empty_like(seq)
    for t in range(T):
        out[t] = normalize_frame(seq[t])
    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("=" * 70)
    print("PSL Word Dataset Builder")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Discover (word, sequence_dir) pairs across all data roots
    print("\nDiscovering sequences...")
    per_class_seqs = {}  # label -> list of seq_dir paths
    source_counts = {}   # (label, root) -> count
    missing_roots = []

    for root in DATA_ROOTS:
        if not os.path.isdir(root):
            missing_roots.append(root)
            continue
        for name in sorted(os.listdir(root)):
            if name in SKIP_ENTRIES:
                continue
            word_dir = os.path.join(root, name)
            if not os.path.isdir(word_dir):
                continue
            seqs = list_sequence_dirs(word_dir)
            if not seqs:
                continue
            per_class_seqs.setdefault(name, []).extend(seqs)
            source_counts[(name, os.path.basename(root))] = len(seqs)

    if missing_roots:
        print(f"[WARN] Missing data roots: {missing_roots}")

    if not per_class_seqs:
        raise RuntimeError("No sequences found. Check DATA_ROOTS paths.")

    labels = sorted(per_class_seqs.keys())
    print(f"\nFound {len(labels)} classes across {len(DATA_ROOTS)} roots:")
    for lbl in labels:
        bits = []
        for root in DATA_ROOTS:
            base = os.path.basename(root)
            c = source_counts.get((lbl, base), 0)
            bits.append(f"{base}={c}")
        total = len(per_class_seqs[lbl])
        print(f"  {lbl:25s} total={total:3d}  ({', '.join(bits)})")

    # 2. Build arrays of sequence paths and integer labels
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    all_seq_paths = []
    all_labels = []
    for lbl in labels:
        for sp in per_class_seqs[lbl]:
            all_seq_paths.append(sp)
            all_labels.append(lbl)
    all_labels_enc = label_encoder.transform(all_labels)

    print(f"\nTotal sequences: {len(all_seq_paths):,}")
    print(f"Classes: {len(labels)}")

    # 3. Stratified 70/15/15 split by sequence.
    #    First carve out test, then split remainder into train/val.
    print("\nSplitting 70/15/15 by sequence (stratified by class)...")
    idx = np.arange(len(all_seq_paths))
    idx_trainval, idx_test = train_test_split(
        idx,
        test_size=TEST_FRACTION,
        random_state=RANDOM_STATE,
        stratify=all_labels_enc,
    )
    # Val fraction relative to trainval so overall split is 70/15/15
    val_rel = VAL_FRACTION / (1.0 - TEST_FRACTION)
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_rel,
        random_state=RANDOM_STATE,
        stratify=all_labels_enc[idx_trainval],
    )
    print(f"  Train: {len(idx_train):,}")
    print(f"  Val:   {len(idx_val):,}")
    print(f"  Test:  {len(idx_test):,}")

    # 4. Load + normalize each sequence into the X tensors
    def materialize(name, indices):
        X = np.zeros((len(indices), T, F), dtype=np.float32)
        y = np.zeros((len(indices),), dtype=np.int64)
        for i, src_idx in enumerate(indices):
            raw = load_sequence(all_seq_paths[src_idx])
            X[i] = normalize_sequence(raw)
            y[i] = all_labels_enc[src_idx]
            if (i + 1) % 500 == 0 or i + 1 == len(indices):
                print(f"  [{name}] {i+1}/{len(indices)}", flush=True)
        return X, y

    print("\nLoading + normalizing sequences...")
    X_train, y_train = materialize("train", idx_train)
    X_val, y_val = materialize("val",   idx_val)
    X_test, y_test = materialize("test",  idx_test)

    # 5. Save outputs
    print(f"\nSaving to {OUTPUT_DIR}/ ...")
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"),   X_val)
    np.save(os.path.join(OUTPUT_DIR, "y_val.npy"),   y_val)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"),  y_test)
    joblib.dump(label_encoder, os.path.join(OUTPUT_DIR, "label_encoder.pkl"))

    config = {
        "T": T,
        "F": F,
        "n_hands": N_HANDS,
        "n_landmarks": N_LANDMARKS,
        "n_coords": N_COORDS,
        "normalization": "per_hand_wrist_centered_maxabs",
        "feature_layout": "[lh_xyz...21][rh_xyz...21]",
        "classes": list(label_encoder.classes_),
        "data_roots": DATA_ROOTS,
        "split": {
            "train": int(len(idx_train)),
            "val": int(len(idx_val)),
            "test": int(len(idx_test)),
        },
    }
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    print("\n" + "=" * 70)
    print("Done.")
    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}  y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}  y_test:  {y_test.shape}")
    print(f"  Classes: {len(label_encoder.classes_)}")
    print("=" * 70)
    print(f"\nNext: python src/training/train_psl_words.py")


if __name__ == "__main__":
    main()
