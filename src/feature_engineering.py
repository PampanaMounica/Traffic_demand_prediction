import pandas as pd

FINAL_FEATURES = [
    'congestion_pressure', 'weather_factor', 'incident_num', 'channel_busy_ratio_pct',
    'occupancy_pct', 'avg_speed_kmph', 'flow_veh_per_hr', 'speed_density_ratio',
    'packet_loss_pct', 'wireless_congestion_intensity', 'queue_length_veh',
    'throughput_per_queued_vehicle', 'hour', 'road_segment_freq',
    'is_rush_hour', 'day_of_week', 'is_weekend'
]

def engineer_features(df, road_freq_map=None):
    """Create time-based features and encode road_segment_id by frequency."""
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_rush_hour'] = df['hour'].apply(lambda h: 1 if (7 <= h <= 10 or 17 <= h <= 20) else 0)

    if road_freq_map is None:
        road_freq_map = df['road_segment_id'].value_counts(normalize=True)
    df['road_segment_freq'] = df['road_segment_id'].map(road_freq_map).fillna(0)

    return df, road_freq_map