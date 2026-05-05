import os
import logging
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