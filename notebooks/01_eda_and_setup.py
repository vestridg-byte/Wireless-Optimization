import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====== CONFIG ======
DATA_PATH = r"data/wifi_rssi.csv"   # <-- rename to your actual file
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ====== LOAD ======
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isna().sum().sort_values(ascending=False).head(20))

# ====== Identify label column automatically (common names) ======
possible_labels = ["label", "Label", "POSITION", "Position", "ROOM", "Room", "zone", "Zone", "LOCATION", "Location", "CLASS", "Class"]
label_col = None
for c in possible_labels:
    if c in df.columns:
        label_col = c
        break

if label_col is None:
    # Fallback: assume last column is label
    label_col = df.columns[-1]
print("Using label column:", label_col)

# ====== Features (RSSI columns) ======
X_cols = [c for c in df.columns if c != label_col]

# Keep only numeric feature columns
X_cols = [c for c in X_cols if pd.api.types.is_numeric_dtype(df[c])]
print("Num feature columns:", len(X_cols))

# Basic stats plot: RSSI distribution
vals = df[X_cols].values.flatten()
vals = vals[~np.isnan(vals)]

plt.figure()
plt.hist(vals, bins=50)
plt.title("RSSI Value Distribution (all AP features)")
plt.xlabel("RSSI")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "rssi_distribution.png"))
plt.show()

# Class balance plot
plt.figure()
df[label_col].value_counts().plot(kind="bar")
plt.title("Samples per Location/Class")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "class_counts.png"))
plt.show()