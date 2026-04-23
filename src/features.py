
import pandas as pd
import numpy as np

def build_master_features(flights_path: str, security_path: str) -> pd.DataFrame:
    """
    Ingests raw flight and security data, resamples to 15-minute windows,
    and applies a 120-minute lead-lag shift to align future flights with current flow.
    """
    print(" Building Feature Pipeline...")
    
    # 1. Load Flights
    flights_columns = [
        'flight_id', 'airline_name', 'airline_code', 'origin', 'destination',
        'scheduled_departure', 'scheduled_arrival', 'actual_departure', 'actual_arrival',
        'aircraft_type', 'aircraft_registration', 'scheduled_capacity', 'actual_capacity',
        'flight_status', 'delay_minutes', 'delay_reason', 'terminal', 'gate',
        'on_tarmac', 'passengers_boarded', 'total_seats', 'gate_assignment_time',
        'crew_ready', 'operational_status', 'on_time_percentage', 'turnaround_time_min',
        'fuel_efficiency_ratio', 'time_of_day', 'day_of_week', 'holiday_flag',
        'season', 'flight_type'
    ]
    flights = pd.read_csv(flights_path, header=None, names=flights_columns, skiprows=1)
    flights['scheduled_departure'] = pd.to_datetime(flights['scheduled_departure'])

    # 2. Process Flights (15-min windows & 120-min shift)
    flights_indexed = flights.set_index('scheduled_departure')
    flights_15m = flights_indexed.resample('15min').agg(
        departing_flights_count=('flight_id', 'count'),
        upcoming_flight_capacity=('scheduled_capacity', 'sum')
    ).reset_index()
    flights_15m.rename(columns={'scheduled_departure': 'time_window'}, inplace=True)
    flights_15m['time_window'] = flights_15m['time_window'] - pd.Timedelta(minutes=120)

    # 3. Load Security
    security_columns = [
        'screening_id', 'pnr_code', 'passenger_id', 'group_size', 'screening_timestamp',
        'flight_scheduled_departure', 'actual_departure', 'screening_status', 'contraband_detected',
        'requires_secondary', 'staff_id', 'lane_id', 'wait_time_minutes', 'is_fast_track',
        'is_staff', 'shift_id', 'throughput_ph', 'queue_length', 'lane_capacity', 'is_peak'
    ]
    security = pd.read_csv(security_path, header=None, names=security_columns, skiprows=1)
    security['screening_timestamp'] = pd.to_datetime(security['screening_timestamp'])

    # 4. Process Security (The Target: Passenger Flow)
    security_indexed = security.set_index('screening_timestamp')
    security_15m = security_indexed.resample('15min').agg(
        target_flow=('passenger_id', 'count') # We use flow instead of wait time due to data variance
    ).reset_index()
    security_15m.rename(columns={'screening_timestamp': 'time_window'}, inplace=True)

    # 5. Merge and Enrich
    master_df = pd.merge(security_15m, flights_15m, on='time_window', how='outer')
    master_df = master_df.sort_values('time_window').fillna(0)
    master_df['hour_of_day'] = master_df['time_window'].dt.hour
    master_df['day_of_week'] = master_df['time_window'].dt.dayofweek

    # Clean up empty gaps created by the shift
    master_df = master_df[master_df['target_flow'] > 0].copy()

    print(f" Master Feature Table ready! Shape: {master_df.shape}")
    return master_df

# If this file is run directly, test the function
if __name__ == "__main__":
    df = build_master_features("data/flights.csv", "data/security_screening.csv")
    print(df.head())