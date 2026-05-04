import os
import mlflow
import numpy as np
import mlflow.sklearn
import mlflow.xgboost
import joblib

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
    features = ["upcoming_flight_capacity", "hour_of_day", "day_of_week"]
    X = df[features]
    
    # Split the data 
    # here shuffle = False to preserve time order and to avoid leaking future information
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size = 0.2, random_state = 42, shuffle = False
    )
    print(f" Training Windows: {len(X_train)} | Testing Windows: {len(X_test)}")
    print(f" Congestion Threshold set at: {congestion_threshold:.0f} passengers/15m")
    
    
    # Mlflow Tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Passenger Flow Prediction")
    
    #1 Logistic Regression
    with mlflow.start_run(run_name = "Logistic Regression"):
        print("\n Training Logistic Regression Classifier...")
        log_reg = LogisticRegression(class_weight = "balanced")
        log_reg.fit(X_train, y_clf_train)
        
        preds_class = log_reg.predict(X_test)
        acc = accuracy_score(y_clf_test, preds_class)
        prec = precision_score(y_clf_test, preds_class, zero_division = 0)
        
        mlflow.log_param("model_type", "Logistic Regression")
        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("precision", float(prec))
        mlflow.sklearn.log_model(log_reg, "logistic regression model")
        
        print(f" Baseline Accuracy: {acc*100:.1f}%")
        print(f" Baseline Precision: {prec*100:.1f}%")
        
    #2 XGBoost Regression
    with mlflow.start_run(run_name = "XGBoost Regression"):
        print("\n Training XGBoost Regressor...")
        
        xgb_model = xgb.XGBRegressor(
            n_estimators = 150,
            learning_rate = 0.05,
            max_depth = 4,
            random_state = 42
        )
        xgb_model.fit(X_train, y_reg_train)
        preds_reg = xgb_model.predict(X_test)
        mae = mean_absolute_error(y_reg_test, preds_reg)
        
        mlflow.log_param("model type", "XGBoost")
        mlflow.log_param("n_estimators", 150)
        mlflow.log_metric("MAE", float(mae))
        mlflow.xgboost.log_model(xgb_model, "xgboost regression model")
        
        print(f" XGBoost MAE: {mae:.2f} passengers / 15m")
        
        #Save the best model
        os.makedirs("../models", exist_ok = True)
        joblib.dump(xgb_model, "../models/best_model.pkl")
        print(" Model saved to models/best_model.pkl")
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_training_pipeline()