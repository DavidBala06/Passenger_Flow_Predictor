# src/ingestion.py
import os

def validate_data_sources(data_dir: str = "../data"): 
    """
    Checks if the required CSV files exist, are accessible, 
    and contain actual data before the pipeline proceeds.
    """
    # These are the 4 core files our features.py script relies on
    required_files = [
        'flights.csv', 
        'security_screening.csv', 
        'baggage.csv', 
        'staff_shifts.csv'
    ]
    
    print(" Starting Data Ingestion & Validation...")
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f" Data directory '{data_dir}' was not found. Please ensure the path is correct.")
    
    # Validate each critical file
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        
        # 1. Check if it exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f" Critical data file missing: {file_path}")
        
        # 2. Check if it's empty (less than 100 bytes is usually just headers)
        if os.path.getsize(file_path) < 100:
            raise ValueError(f" Data file {file_path} is empty or too small to be valid.")
            
        print(f" {file} located and validated!")
        
    print(" All critical data files are present and valid. Ready for feature engineering!")
    return True

if __name__ == "__main__":
    # Ensure correct working directory context if run directly
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        validate_data_sources("../data")
    except Exception as e:
        print(f"\n INGESTION FAILED: {e}")