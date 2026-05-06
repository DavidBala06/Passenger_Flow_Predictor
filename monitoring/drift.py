import logging
from scipy.stats import ks_2samp

#set up logging
logging.basicConfig(
    level = logging.INFO,
    format="%(asctime)s - DRIFT MONITOR - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def check_feature_drift(reference_data, current_data, feature_name = "upcoming_flight_capacity"):
    # Compares the live api data to the training data
    if len(current_data) < 10:
        logger.info("Not enough data to check drift yet")
        return False
    statistic, p_value = ks_2samp(reference_data, current_data)
    #statistic method check if p_value < critical value(arbitrarily set at 0.05) to determine if distributions are different:
    if p_value < 0.05: 
        logger.warning(f"Drift detected in feature: {feature_name}")
        return True
    else:
        logger.info(f"No drift detected in feature: {feature_name}")
        return False


import pandas as pd

def run_automated_check():
    # 1. Load the baseline saved by train.py
    baseline = pd.read_csv("baseline_data.csv")['upcoming_flight_capacity']
    
    # 2. Load the recent requests saved by the API
    live_data = pd.read_csv("live_api_data.csv", names=['upcoming_flight_capacity'])
    
    # 3. Only keep the last 500 requests (we only care about RECENT data)
    recent_live_data = live_data.tail(500)['upcoming_flight_capacity']
    
    # 4. Compare them!
    check_feature_drift(baseline, recent_live_data)