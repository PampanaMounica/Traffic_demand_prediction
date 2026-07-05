# Dataset Documentation

## Source
Custom synthetic VANET (Vehicular Ad-hoc Network) traffic dataset, designed to
realistically simulate urban traffic congestion scenarios. Not collected from
real-world sensors.

## File
`vanet_traffic_data.csv` — 195,714 rows × 27 columns

## Time Range
2025-09-28 to 2025-09-30 (approx. 2.5 days, high-frequency sampling across 500 road segments)

## Target Variable
`label` — 4-class traffic congestion state:
- Free-flow (57,812 records)
- Moderate (54,018 records)
- Heavy (47,440 records)
- Gridlock (36,444 records)

## Column Descriptions

| Column | Type | Description |
|---|---|---|
| timestamp | datetime | Record timestamp |
| road_segment_id | categorical | Unique ID for one of 500 road segments |
| avg_speed_kmph | float | Average vehicle speed (km/h) |
| density_veh_per_km | float | Vehicle density per km |
| avg_wait_time_s | float | Average wait time (seconds) |
| occupancy_pct | float | Road occupancy percentage |
| flow_veh_per_hr | float | Traffic flow (vehicles/hour) |
| queue_length_veh | float | Queue length (vehicles) |
| avg_accel_ms2 | float | Average acceleration (m/s²) |
| heading_deg | float | Vehicle heading (degrees) |
| signal_state_num | float | Numeric traffic signal state |
| incident_num | float | Number/severity of incidents |
| temp_c | float | Temperature (°C) |
| visibility_km | float | Visibility (km) |
| rain_intensity_mmph | float | Rainfall intensity (mm/hr) |
| weather_factor | float | Composite weather impact score |
| channel_busy_ratio_pct | float | VANET wireless channel busy ratio |
| msg_rate_hz | float | Message transmission rate (Hz) |
| avg_comm_delay_ms | float | Average communication delay (ms) |
| rssi_dbm | float | Received signal strength (dBm) |
| packet_loss_pct | float | Packet loss percentage |
| speed_density_ratio | float | Derived: speed/density ratio |
| congestion_pressure | float | Derived: congestion pressure score |
| wireless_congestion_intensity | float | Derived: wireless congestion score |
| throughput_per_queued_vehicle | float | Derived: throughput per queued vehicle |
| acceleration_directionality | float | Derived: directional acceleration score |
| label | categorical | Target: traffic congestion class |

## Data Quality Notes
- 0 missing values, 0 duplicate rows
- Some derived features contained small negative values near zero
  (`queue_length_veh`, `congestion_pressure`, `incident_num`) — clipped to 0
  during preprocessing as physically implausible simulation noise
- Features exhibit high multicollinearity (VIF up to 3800+) due to synthetic
  generation from shared underlying formulas — addressed via feature selection
  (see project report for full analysis)