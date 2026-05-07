import logging
import os
import sys
from scipy.stats import ks_2samp
import pandas as pd

#set up logging
logging.basicConfig(
    level=logging.INFO,
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


def run_automated_check():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    baseline_path = os.path.join(script_dir, "baseline_data.csv")
    live_path = os.path.join(script_dir, "live_api_data.csv")

    if os.path.exists(baseline_path):
        baseline = pd.read_csv(baseline_path)["upcoming_flight_capacity"]
        logger.info(f"Loaded baseline from {baseline_path}")
    else:
        data_dir = os.path.join(project_root, "data")
        sys.path.insert(0, os.path.join(project_root, "src"))
        try:
            from features import build_master_features
            baseline = build_master_features(data_dir)["upcoming_flight_capacity"]
            logger.info("Derived baseline from raw data because baseline_data.csv was not found.")
        except Exception as err:
            logger.error(
                "Baseline file not found and failed to derive baseline from raw data. "
                "Create baseline_data.csv in monitoring/ or ensure data/ exists."
            )
            raise FileNotFoundError(f"Baseline data unavailable: {err}") from err

    if not os.path.exists(live_path):
        logger.error(f"Live API data file not found: {live_path}. Skipping drift check.")
        return

    live_data = pd.read_csv(live_path, names=["upcoming_flight_capacity"])
    recent_live_data = live_data.tail(500)["upcoming_flight_capacity"]
    check_feature_drift(baseline, recent_live_data)


if __name__ == "__main__":
    run_automated_check()
