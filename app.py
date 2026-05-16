import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from channel_selector import select_best_channel

st.set_page_config(
    page_title="AI-Driven Wireless Networking",
    layout="wide"
)

st.title("AI-Driven Wireless Networking")
st.sidebar.title("Project Info")
st.sidebar.write("AI-driven wireless channel selection using RSSI, SNR, interference, and load.")
st.sidebar.write("Model: Random Forest Regressor")
st.sidebar.write("Goal: Maximize throughput while reducing interference and channel-switch thrashing.")

st.write(
    "This demo simulates Wi-Fi/cellular channel conditions and uses an AI model "
    "to select the best channel at each time step."
)

CHANNELS = [1, 2, 3, 4, 5]
TIME_STEPS = 50

previous_channel = None
selected_channels = []
throughputs = []
switch_count = 0
all_results = []

for t in range(TIME_STEPS):
    channel_data = []

    for channel in CHANNELS:
        rssi = np.random.uniform(-90, -35)
        snr = np.random.uniform(5, 40)
        interference = np.random.uniform(0, 1)
        load = np.random.uniform(0, 1)

        if snr > 30:
            quality = "Excellent"
        elif snr > 20:
            quality = "Good"
        elif snr > 10:
            quality = "Fair"
        else:
            quality = "Poor"

        channel_data.append({
            "channel": channel,
            "rssi": round(rssi, 2),
            "snr": round(snr, 2),
            "interference": round(interference, 3),
            "load": round(load, 3),
            "quality": quality
        })

    best_channel, results = select_best_channel(
        channel_data,
        previous_channel
    )

    selected_row = results[results["channel"] == best_channel].iloc[0]
    decision_reason = f"""
        AI selected Channel {best_channel} because it had:

        - High predicted throughput
        - Lower interference
        - Acceptable traffic load
        - Reduced switching cost
        """

    if previous_channel is not None and best_channel != previous_channel:
        switch_count += 1

    previous_channel = best_channel
    selected_channels.append(best_channel)
    throughputs.append(selected_row["predicted_throughput"])

    results["time_step"] = t
    all_results.append(results)

final_df = pd.concat(all_results, ignore_index=True)

col1, col2, col3 = st.columns(3)

col1.metric("Total Time Steps", TIME_STEPS)
col2.metric("Channel Switches", switch_count)
col3.metric("Average Predicted Throughput", f"{np.mean(throughputs):.2f} Mbps")

st.subheader("AI Decision Explanation")
st.info(decision_reason)

st.subheader("Latest Channel Decision Table")
st.dataframe(
    final_df[final_df["time_step"] == TIME_STEPS - 1][
        ["channel", "rssi", "snr", "interference", "load", "predicted_throughput", "score"]
    ]
)

st.subheader("Network Performance Analytics")

colA, colB, colC = st.columns(3)

# ---------------- GRAPH 1 ----------------

with colA:

    fig1, ax1 = plt.subplots(figsize=(3.5, 2))

    ax1.plot(
        range(TIME_STEPS),
        selected_channels,
        marker="o",
        markersize=3
    )

    ax1.set_title("Selected Channel", fontsize=9)
    ax1.set_xlabel("Time", fontsize=7)
    ax1.set_ylabel("Channel", fontsize=7)

    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

    st.pyplot(fig1)

# ---------------- GRAPH 2 ----------------

with colB:

    fig2, ax2 = plt.subplots(figsize=(3.5, 2))

    ax2.plot(
        range(TIME_STEPS),
        throughputs,
        marker="o",
        markersize=3
    )

    ax2.set_title("Predicted Throughput", fontsize=9)
    ax2.set_xlabel("Time", fontsize=7)
    ax2.set_ylabel("Mbps", fontsize=7)

    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

    st.pyplot(fig2)

# ---------------- GRAPH 3 ----------------

with colC:

    avg_tp = final_df.groupby("channel")["predicted_throughput"].mean()

    fig3, ax3 = plt.subplots(figsize=(3.5, 2))

    ax3.bar(
        avg_tp.index.astype(str),
        avg_tp.values
    )

    ax3.set_title("Channel Comparison", fontsize=9)
    ax3.set_xlabel("Channel", fontsize=7)
    ax3.set_ylabel("Avg TP", fontsize=7)

    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

    st.pyplot(fig3)

st.subheader("All Simulation Results")
st.dataframe(final_df)