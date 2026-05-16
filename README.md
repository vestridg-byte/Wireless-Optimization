# AI-Driven Wireless Networking

## Overview

AI-Driven Wireless Networking is an intelligent channel selection system that uses machine learning to optimize wireless communication performance in dynamic network environments.

The system continuously analyzes wireless network conditions such as:

- RSSI (Received Signal Strength Indicator)
- SNR (Signal-to-Noise Ratio)
- Interference
- Network Load

Using these parameters, the AI model predicts network throughput and dynamically selects the best wireless channel while minimizing interference and reducing unnecessary channel switching (channel thrashing).

---

# Problem Statement

Traditional wireless networks often suffer from:

- High interference
- Congested communication channels
- Frequent unstable channel switching
- Reduced throughput
- Poor adaptive decision-making

Static channel allocation methods cannot efficiently respond to rapidly changing wireless conditions.

This project introduces an AI-driven adaptive channel selection mechanism that intelligently predicts and selects the optimal communication channel in real time.

---

# Objectives

- Predict wireless network throughput using machine learning
- Dynamically select the optimal channel
- Reduce interference impact
- Minimize channel-switch thrashing
- Improve overall network efficiency
- Simulate adaptive wireless resource allocation

---

# System Architecture

```text
Wireless Environment
        ↓
Channel Measurements
(RSSI, SNR, Interference, Load)
        ↓
AI Prediction Model
(Random Forest Regressor)
        ↓
Decision Engine
(Throughput + Penalty Logic)
        ↓
Best Channel Selection
        ↓
Performance Monitoring Dashboard
```

---

# Features

## AI-Based Throughput Prediction

The system predicts expected throughput for each wireless channel using a trained Random Forest Regressor.

---

## Dynamic Channel Selection

The AI engine evaluates all available channels and selects the channel with the highest performance score.

---

## Interference-Aware Optimization

The decision engine penalizes channels with high interference to improve communication stability.

---

## Channel Switching Penalty

A switching penalty mechanism is introduced to reduce unnecessary frequent channel changes.

---

## Real-Time Simulation Dashboard

The Streamlit dashboard provides:

- Live wireless condition simulation
- AI decision explanation
- Throughput analytics
- Channel comparison visualization
- Network performance monitoring

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive dashboard |
| Scikit-learn | Machine learning model |
| Random Forest Regressor | Throughput prediction |
| Pandas | Data processing |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |

---

# Machine Learning Model

## Model Used

Random Forest Regressor

---

## Input Features

- Channel Number
- RSSI
- SNR
- Interference
- Load

---

## Target Variable

- Throughput

---

# Decision Logic

The channel selection score is calculated using:

```text
Score =
Predicted Throughput
- Interference Penalty
- Channel Switching Penalty
```

The channel with the highest score is selected by the AI system.

---

# Dashboard Analytics

The dashboard visualizes:

- Selected Channel Over Time
- Predicted Throughput Over Time
- Channel Performance Comparison
- AI Decision Explanation
- Wireless Network Simulation Results

---

# Project Structure

```text
AI-Driven-Wireless-Networking/
│
├── app.py
├── channel_selector.py
├── train_model.py
├── generate_dataset.py
├── wireless_dataset.csv
├── throughput_model.pkl
├── requirements.txt
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone <repository-link>
cd AI-Driven-Wireless-Networking
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run the Project

## Step 1 — Generate Dataset

```bash
python generate_dataset.py
```

---

## Step 2 — Train Model

```bash
python train_model.py
```

---

## Step 3 — Launch Dashboard

```bash
streamlit run app.py
```

---

# Sample Output

The system dynamically simulates wireless conditions and performs intelligent channel selection using AI-driven throughput prediction.

The dashboard displays:

- Adaptive channel selection
- Throughput variation
- Interference-aware optimization
- Real-time wireless analytics

---

# Future Improvements

- Reinforcement Learning Integration
- Real Wireless Dataset Integration
- 5G/6G Resource Block Optimization
- IoT Device Optimization
- Edge AI Deployment
- ns-3 Network Simulation Integration
- Deep Learning-Based Prediction Models

---

# Applications

- Smart Wireless Networks
- 5G/6G Communication Systems
- IoT Networks
- Adaptive Wi-Fi Optimization
- Intelligent Resource Allocation
- AI-Based Telecom Systems

---

# Conclusion

This project demonstrates how artificial intelligence can improve wireless networking through adaptive channel selection and throughput optimization.

By combining machine learning prediction with interference-aware decision-making, the system dynamically adapts to changing wireless conditions and improves overall communication efficiency.

---

# Authors

- Rushi Lunagariya
- Ishaan Dange
- Vincent Estridge  

---

# License

This project is developed for academic and educational purposes.