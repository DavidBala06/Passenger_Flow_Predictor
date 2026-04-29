import os
import mlflow
import numpy as np
import mlflow.sklearn
import mlflow.xgboost

#models
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import xgboost as xgb

# metrics
from sklearn.metrics import accuracy_score, mean_absolute_error, precision_score

# import feature pipelines
from features import build_master_features

def run_training_pipeline():
    df = build_master_features("../data")
    
    # Create targets 
    # Target 1  - exact number of passengers (for xgboost regression)
    y_reg = df["target_flow"]
    
    # Target 2 - congestion flag (for logistic regression classification)
    # Let's say if flow is in the top 20% of busiest times, it s a Congestion flag
    congestion_threshold = df["target_flow"].quantile(0.8)
    y_clf = (df["target_flow"] >= congestion_threshold).astype(int)
    
    #features
    features = ["upcoming_flight_capacity", "hour of day", "day_of_week"]
    X = df[features]
    
    # Split the data 
    # here shuffle = False to preserve time order and to avoid leaking future information
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size = 0.2, random_state = 42, shuffle = False
    )
    print(f" Training Windows: {len(X_train)} | Testing Windows: {len(X_test)}")
    print(f" Congestion Threshold set at: {congestion_threshold:.0f} passengers/15m")
    
    
    # Mlflow Tracking
    