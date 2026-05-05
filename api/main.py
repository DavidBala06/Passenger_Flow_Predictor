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


