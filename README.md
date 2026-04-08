# ✈️ Passenger Flow Predictor & Smart Staffing System

## 📌 Overview

Airports often face unpredictable spikes in passenger volume, leading to long queues at security and check-in. This results in poor passenger experience, operational stress, and delayed responses from staff.

This project aims to **predict queue wait times 60–120 minutes in advance** and provide **actionable staffing recommendations** to mitigate congestion before it happens.

It is designed as a **production-ready end-to-end MLOps system**, not just a machine learning model.

---

## 🎯 Objectives

* Predict future queue wait times using historical and operational data
* Detect upcoming congestion peaks
* Recommend proactive actions (e.g., opening lanes, reallocating staff)
* Build a fully reproducible and deployable ML pipeline

---

## 🧠 Key Features

* ⏱ Time-series forecasting of queue wait times
* ⚙️ Automated data ingestion and preprocessing
* 🧩 Reusable feature engineering pipeline
* 📊 Model comparison (baseline vs optimized)
* 🧪 Experiment tracking with MLflow
* 🚀 REST API for real-time predictions
* 🐳 Dockerized deployment
* 📜 Logging for traceability
* 📉 Basic drift detection for monitoring

---

## 🏗️ System Architecture

```
Data Sources → Data Ingestion → Feature Engineering → Model Training
        → MLflow Tracking → Model Registry → FastAPI API
        → Logging & Monitoring (Drift Detection)
```

---

## 📂 Project Structure

```
project/
│
├── data/                  # Raw and processed datasets
├── notebooks/             # Exploratory analysis
├── src/
│   ├── ingestion.py       # Data ingestion & validation
│   ├── features.py        # Feature engineering pipeline
│   ├── train.py           # Model training
│   ├── evaluate.py        # Model evaluation
│   ├── predict.py         # Inference logic
│
├── api/
│   └── main.py            # FastAPI app
│
├── monitoring/
│   └── drift.py           # Drift detection logic
│
├── models/                # Saved models
├── mlruns/                # MLflow tracking data
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

The project uses a multi-table airport dataset including:

* Flight schedules
* Passenger information
* Security screening events
* Staff shift data

### ⚠️ Important

The dataset does not directly include queue wait times, so the target variable is **engineered** from available timestamps and flow data.

---

## 🔧 Pipelines

### 1️⃣ Data Ingestion Pipeline

* Loads raw data from multiple sources
* Validates schema and data quality
* Outputs clean, structured data

---

### 2️⃣ Feature Engineering Pipeline

* Time-based features (hour, day, seasonality)
* Rolling statistics (e.g., last 30–60 min flow)
* Future flight load (next 2 hours)
* Staff capacity features

Ensures consistency between training and inference.

---

## 🤖 Models

### Baseline

* Logistic Regression (classification of congestion risk)

### Advanced

* XGBoost (regression for wait time prediction)

---

## 🧪 Experiment Tracking (MLflow)

Tracks:

* Model parameters
* Evaluation metrics (RMSE, MAE, etc.)
* Model artifacts

Enables reproducibility and comparison across experiments.

---

## 📈 Model Evaluation

* Time-based cross-validation
* Metrics:

  * RMSE / MAE (regression)
  * Precision / Recall (classification)
* Optional calibration analysis

---

## 🚀 API (FastAPI)

### Endpoint

```
POST /predict
```

### Input

* Flight data
* Current passenger flow
* Staff availability

### Output

```json
{
  "predicted_wait_time": 32,
  "recommended_action": "open_2_lanes"
}
```

---

## 🐳 Docker

The entire application is containerized to ensure:

* Environment consistency
* Easy deployment
* Portability across systems

---

## 📜 Logging

Logs include:

* Incoming requests
* Predictions
* Errors and exceptions

Used for debugging and system observability.

---

## 📉 Monitoring (Drift Detection)

Tracks:

* Feature distribution shifts
* Prediction distribution changes

Simple statistical tests (e.g., KS test) trigger alerts when drift is detected.

---

## 🧠 Decision Engine

Transforms predictions into actions:

Example rules:

* If wait time > threshold → open additional lanes
* If rapid increase detected → reassign staff
* If peak predicted → trigger early alert

---

## 🚀 Getting Started

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Run training

```
python src/train.py
```

### 3. Start API

```
uvicorn api.main:app --reload
```

### 4. Run with Docker

```
docker build -t passenger-flow .
docker run -p 8000:8000 passenger-flow
```

---

## 📌 Future Improvements

* Real-time streaming data integration
* Computer vision for live passenger counting
* Reinforcement learning for staffing optimization
* Advanced monitoring dashboards
* Multi-zone airport modeling

---

## 🏁 Conclusion

This project demonstrates how to move from a business problem to a **production-ready machine learning system**, combining data engineering, modeling, deployment, and monitoring into a cohesive solution.

It is designed to reflect real-world challenges in operations, scalability, and decision-making.
