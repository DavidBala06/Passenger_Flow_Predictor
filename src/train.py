import os
import mlflow
import numpy as np
import mlflow.sklearn
import mlflow.xgboost

#models
from sklearn.linear_model import LogisticRegression
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
    