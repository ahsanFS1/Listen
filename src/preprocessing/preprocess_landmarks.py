# src/preprocessing/preprocess_landmarks.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# ---------------------------------------------------------------------
# ⚙️ Configuration
# ---------------------------------------------------------------------
INPUT_CSV = "data/landmarks_fast.csv"
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# 🧠 Load and clean data
# ---------------------------------------------------------------------
df = pd.read_csv(INPUT_CSV)
print("Original shape:", df.shape)

# Drop 'nothing' (no hands detected)
df = df[df["label"] != "nothing"].reset_index(drop=True)
print("After removing 'nothing':", df.shape)

# Separate features and labels
X = df.drop("label", axis=1)
y = df["label"]

# ---------------------------------------------------------------------
# 🔢 Encode labels (A, B, C → 0, 1, 2, ...)
# ---------------------------------------------------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print("Classes:", list(label_encoder.classes_))
print("Total classes:", len(label_encoder.classes_))

# ---------------------------------------------------------------------
# 📏 Normalize x,y coordinates (mean=0, std=1)
# ---------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------------------
# ✂️ Split into training and testing sets
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# ---------------------------------------------------------------------
# 💾 Save processed data & encoders
# ---------------------------------------------------------------------
pd.DataFrame(X_train).to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
pd.DataFrame(X_test).to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)
pd.DataFrame(y_train).to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
pd.DataFrame(y_test).to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)

joblib.dump(label_encoder, os.path.join(OUTPUT_DIR, "label_encoder.pkl"))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

print("\n✅ Preprocessing complete! Files saved to:", OUTPUT_DIR)
