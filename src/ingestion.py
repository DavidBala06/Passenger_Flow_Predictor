import os
import pandas as pd

def validate_data_sources(data_dir: str = "../data"): 
    #Check if the required data files exit and ready to use
    
    # These are the 4 core files our features.py script relies on
    required_files = [
        'flights.csv', 
        'security_screening.csv', 
        'baggage.csv', 
        'staff_shifts.csv'
    ]
    print("Starting data inspection and validation ...")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} it was not found. Please ensure the path is correct")
    
    #validate each file
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Critical data file {file_path} is missing.")
        
        #check if file is not empty
        if os.path.getsize(file_path) < 100: #arbitrary small size to catch empty files
            raise ValueError(f"Data file {file_path} is empty or too small to be valid")
        print(f"{file_path} located and validated!")
        
    print("All critical data files are present and valid. Ready for feature engineering!")
    return True

#small test to check if the function works
if __name__ == "__main__":
    try:
        validate_data_sources("../data")
    except Exception as e:
        print(f"Ingestion failed: {e}")