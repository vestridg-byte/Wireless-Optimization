# Dataset

**Name:** WiFi Received Signal Strength (RSSI) Dataset  
**Source (Kaggle):** https://www.kaggle.com/datasets/sanchukanirupama/wifi-received-signal-strength-rssi-dataset

## How to download
1. Download the dataset from Kaggle (ZIP).
2. Extract the CSV into this folder: `data/`

**Expected file:** a CSV placed at `data/<dataset>.csv`  
(We do not commit the raw dataset file to GitHub.)

## Columns used in this project
**Targets (position):**
- `X` : x-coordinate of device location
- `Y` : y-coordinate of device location

**Features (AP signal strengths):**
- `RSS-A`
- `RSS-B`
- `RSS-C`

## Notes
- RSSI values are treated as numeric features.
- We standardize features using the training split only.
- Goal: select the smallest subset of RSS features that minimizes mean position error.