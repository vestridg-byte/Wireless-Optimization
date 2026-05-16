import pandas as pd
import joblib

model = joblib.load("throughput_model.pkl")

SWITCHING_PENALTY = 5
INTERFERENCE_PENALTY = 20

def select_best_channel(channel_data, previous_channel=None):
    """
    channel_data must contain:
    channel, rssi, snr, interference, load
    """

    df = pd.DataFrame(channel_data)

    features = df[["channel", "rssi", "snr", "interference", "load"]]
    df["predicted_throughput"] = model.predict(features)

    scores = []

    for _, row in df.iterrows():
        score = row["predicted_throughput"]

        score -= INTERFERENCE_PENALTY * row["interference"]

        if previous_channel is not None and row["channel"] != previous_channel:
            score -= SWITCHING_PENALTY

        scores.append(score)

    df["score"] = scores

    best_row = df.loc[df["score"].idxmax()]

    return int(best_row["channel"]), df


if __name__ == "__main__":
    sample_data = [
        {"channel": 1, "rssi": -60, "snr": 25, "interference": 0.2, "load": 0.4},
        {"channel": 2, "rssi": -55, "snr": 28, "interference": 0.6, "load": 0.3},
        {"channel": 3, "rssi": -70, "snr": 18, "interference": 0.1, "load": 0.5},
        {"channel": 4, "rssi": -50, "snr": 30, "interference": 0.3, "load": 0.6},
        {"channel": 5, "rssi": -65, "snr": 22, "interference": 0.4, "load": 0.2},
    ]

    best_channel, results = select_best_channel(
        sample_data,
        previous_channel=1
    )

    print("AI Selected Channel:", best_channel)
    print(results)