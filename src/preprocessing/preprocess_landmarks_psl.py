"""
Preprocess PSL (Pakistan Sign Language) landmarks for training.
Loads landmarks, encodes labels, normalizes features, and splits into train/test sets.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# ---------------------------------------------------------------------
# ⚙️ Configuration
# ---------------------------------------------------------------------
INPUT_CSV = "data/psl_landmarks.csv"
OUTPUT_DIR = "data/psl_processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# 🧠 Load and clean data
# ---------------------------------------------------------------------
print("=" * 70)
print("🧠 PSL Landmark Preprocessing")
print("=" * 70)
print(f"\nInput: {INPUT_CSV}")
print(f"Output: {OUTPUT_DIR}\n")
print("=" * 70 + "\n")

df = pd.read_csv(INPUT_CSV)
print(f"Original shape: {df.shape}")
print(f"Total samples: {len(df):,}")

# Check for any 'nothing' or invalid labels (if any exist)
if "nothing" in df["label"].values:
    df = df[df["label"] != "nothing"].reset_index(drop=True)
    print(f"After removing 'nothing': {df.shape}")

# Separate features and labels
X = df.drop("label", axis=1)
y = df["label"]

# ---------------------------------------------------------------------
# 🔢 Encode labels (Urdu alphabets → 0, 1, 2, ...)
# ---------------------------------------------------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"\n📋 Label Encoding:")
print(f"Total classes: {len(label_encoder.classes_)}")
print(f"\nUrdu Alphabet Classes:")
for idx, label in enumerate(sorted(label_encoder.classes_)):
    count = (y == label).sum()
    print(f"  {idx:2d}. {label:20s} - {count:5,} samples")

# ---------------------------------------------------------------------
# 📏 Normalize x,y coordinates (mean=0, std=1)
# ---------------------------------------------------------------------
print(f"\n📏 Normalizing landmark coordinates...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------------------
# ✂️ Split into training and testing sets (80/20 split)
# ---------------------------------------------------------------------
print(f"\n✂️  Splitting dataset (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training samples: {X_train.shape[0]:,}")
print(f"Testing samples: {X_test.shape[0]:,}")
print(f"Feature dimensions: {X_train.shape[1]} (21 landmarks × 2 coordinates)")

# ---------------------------------------------------------------------
# 💾 Save processed data & encoders
# ---------------------------------------------------------------------
print(f"\n💾 Saving processed data...")

pd.DataFrame(X_train).to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
pd.DataFrame(X_test).to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)
pd.DataFrame(y_train, columns=["label"]).to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
pd.DataFrame(y_test, columns=["label"]).to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)

joblib.dump(label_encoder, os.path.join(OUTPUT_DIR, "label_encoder.pkl"))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

print(f"\n✅ Files saved:")
print(f"  • X_train.csv ({X_train.shape[0]:,} samples)")
print(f"  • X_test.csv ({X_test.shape[0]:,} samples)")
print(f"  • y_train.csv")
print(f"  • y_test.csv")
print(f"  • label_encoder.pkl ({len(label_encoder.classes_)} classes)")
print(f"  • scaler.pkl")

print("\n" + "=" * 70)
print("✅ PSL preprocessing complete!")
print("=" * 70)
print(f"\nNext step: python src/training/train_psl.py")
print("=" * 70)

