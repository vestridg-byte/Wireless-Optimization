# src/train_model.py
"""
Train throughput prediction model for AI-driven channel selection.
Reads data/wireless_dataset.csv, trains a RandomForestRegressor,
and saves models/throughput_model.pkl.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def metrics(y_true, y_pred):
    """Compute MAE, RMSE, R^2 without sklearn.metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    return mae, rmse, r2


def main():
    data_path = os.path.join("data", "wireless_dataset.csv")
    model_path = os.path.join("models", "throughput_model.pkl")
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv(data_path)

    feature_cols = ["channel", "rssi", "snr", "interference", "load"]
    target_col = "throughput"

    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=695
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=695,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae, rmse, r2 = metrics(y_test, preds)

    print("Saved model:", model_path)
    print("Test MAE:", round(mae, 3))
    print("Test RMSE:", round(rmse, 3))
    print("Test R^2:", round(r2, 3))

    joblib.dump(model, model_path)


if __name__ == "__main__":
    main()