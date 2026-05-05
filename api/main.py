import os
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

#Set up logging
logging.basicConfig(
    level = logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


app = FastAPI(title=" Passenger Flow Predictor API")

#LOAD THE SAVED MODEL
# Using dynamic pathing so it works whether run from root or api folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"failed to load model: {e}")
    model = None

#Define input schema
class PredictionRequest(BaseModel):
    upcoming_flight_capacity: int
    hour_of_day: int
    day_of_week: int
    #optional fields for future use
    current_passenger_flow: int = 0
    staff_availability: int = 0
    
#Define an endpoint for nice visualization
@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Welcome to the Passenger Flow Predictor API. Use the /predict endpoint to get wait time predictions and lane recommendations.",
    }
    
@app.post("/predict")
def predict_flow(request: PredictionRequest):
    if model is None:
        logger.error("Model is not available for predictions")
        raise HTTPException(status_code = 500, detail = "Model not available")
    logger.info(f"Incoming request: {request.model_dump()}")
    
    try:
        #format data for xgboost wxactly as it was trained
        input_data = pd.DataFrame([{
            "upcoming_flight_capacity": request.upcoming_flight_capacity,
            "hour_of_day": request.hour_of_day,
            "day_of_week": request.day_of_week
        }])
        
        #predict the passenger flow
        predicted_flow = model.predict(input_data)[0]
        #convert flow to wait time (assuming ~5 pax clear per minute per lane)
        predicted_wait_time = int(predicted_flow / 5)
        
        #Decision engine
        if predicted_wait_time > 45:
            action = "open_3_lanes"
        elif predicted_wait_time > 30:
            action = "open_2_lanes"
        else:
            action = "Maintain_current_lanes"
        
        response = {
            "predicted_wait_time_minutes": predicted_wait_time,
            "recommended_action": action
        }
        #log sucessful prediction
        logger.info(f"Prediction successful: {response}")
        return response
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code = 500, detail = "Prediction failed")
            