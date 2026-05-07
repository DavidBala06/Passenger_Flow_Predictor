import os
import mlflow
import numpy as np
import mlflow.sklearn
import mlflow.xgboost
import joblib
from sklearn.model_selection import RandomizedSearchCV

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
        
    # 2 XGBoost Hyperparameter Tuning
    with mlflow.start_run(run_name="XGBoost_Hyperparameter_Tuning"):
        print("\n Running Hyperparameter Tuning for XGBoost (This might take a minute)...")
        
        #  Define the parameters you want to test
        param_grid = {
            'n_estimators': [100, 150, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [4, 6, 8, 10],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        #  Set up the basic model
        base_xgb = xgb.XGBRegressor(random_state=42)
        
        #  Set up the Random Search engine
        # n_iter=10 means "Try 10 random combinations from the grid above"
        random_search = RandomizedSearchCV(
            estimator=base_xgb,
            param_distributions=param_grid,
            n_iter=10, 
            scoring='neg_mean_absolute_error',
            cv=3, # 3-fold cross validation
            verbose=1,
            random_state=42
        )
        
        #  Start the test!
        random_search.fit(X_train, y_reg_train)
        
        #  Extract the absolute best model from the test
        best_xgb = random_search.best_estimator_
        print(f"\n Best Parameters Found: {random_search.best_params_}")
        
        # Test the best model on our future data
        preds_reg = best_xgb.predict(X_test)
        mae = mean_absolute_error(y_reg_test, preds_reg)
        
        # Log everything to MLflow
        mlflow.log_param("model_type", "XGBoost_Tuned")
        mlflow.log_params(random_search.best_params_) # Logs the winning parameters!
        mlflow.log_metric("MAE", mae)
        mlflow.xgboost.log_model(best_xgb, "advanced_model")
        
        print(f" Tuned XGBoost MAE: Off by {mae:.2f} passengers per 15-min")

        # Save the absolute best model
        os.makedirs("../models", exist_ok=True)
        joblib.dump(best_xgb, "../models/best_model.pkl")
        print(" Best model saved to models/best_model.pkl")
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_training_pipeline()