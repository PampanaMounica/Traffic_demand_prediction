import pandas as pd

def load_and_clean_data(filepath):
    """Load raw CSV and apply basic cleaning (clip invalid negatives)."""
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Clip physically-impossible negative values
    df['queue_length_veh'] = df['queue_length_veh'].clip(lower=0)
    df['congestion_pressure'] = df['congestion_pressure'].clip(lower=0)
    df['incident_num'] = df['incident_num'].clip(lower=0)
    
    return df