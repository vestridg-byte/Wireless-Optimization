import numpy as np
import pandas as pd

np.random.seed(42)

NUM_TIME_STEPS = 200
CHANNELS = [1, 2, 3, 4, 5]

data = []

for t in range(NUM_TIME_STEPS):
    for channel in CHANNELS:
        # Simulated wireless conditions
        rssi = np.random.uniform(-90, -35)          # Signal strength in dBm
        snr = np.random.uniform(5, 40)              # Signal-to-noise ratio in dB
        interference = np.random.uniform(0, 1)      # 0 = low, 1 = high
        load = np.random.uniform(0, 1)              # 0 = low traffic, 1 = congested

        # Simulated throughput formula
        throughput = (
            2.0 * snr
            + 0.5 * (rssi + 100)
            - 30 * interference
            - 25 * load
            + np.random.normal(0, 3)
        )

        throughput = max(throughput, 1)

        data.append({
            "time_step": t,
            "channel": channel,
            "rssi": round(rssi, 2),
            "snr": round(snr, 2),
            "interference": round(interference, 3),
            "load": round(load, 3),
            "throughput": round(throughput, 2)
        })

df = pd.DataFrame(data)
df.to_csv("data/wireless_dataset.csv", index=False)

print("Dataset generated successfully!")
print(df.head())
print("\nDataset shape:", df.shape)