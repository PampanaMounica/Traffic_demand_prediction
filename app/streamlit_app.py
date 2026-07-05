import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from predict import TrafficDemandPredictor

st.set_page_config(page_title="Traffic Demand Prediction", layout="wide")

st.title("🚦 Traffic Demand Prediction")
st.markdown("Predicts traffic congestion level using LightGBM + XGBoost ensemble.")

@st.cache_resource
def load_predictor():
    return TrafficDemandPredictor(model_dir='../models/')

predictor = load_predictor()

st.sidebar.header("Input Traffic Parameters")

timestamp = st.sidebar.text_input("Timestamp (YYYY-MM-DDTHH:MM:SS)", "2025-09-29T08:30:00")
road_segment_id = st.sidebar.text_input("Road Segment ID", "S090")
avg_speed_kmph = st.sidebar.number_input("Avg Speed (km/h)", 0.0, 150.0, 43.0)
density_veh_per_km = st.sidebar.number_input("Density (veh/km)", 0.0, 200.0, 25.0)
avg_wait_time_s = st.sidebar.number_input("Avg Wait Time (s)", 0.0, 300.0, 20.0)
occupancy_pct = st.sidebar.number_input("Occupancy (%)", 0.0, 100.0, 40.0)
flow_veh_per_hr = st.sidebar.number_input("Flow (veh/hr)", 0.0, 3000.0, 1450.0)
queue_length_veh = st.sidebar.number_input("Queue Length (veh)", 0.0, 100.0, 7.0)
incident_num = st.sidebar.number_input("Incident Number", 0.0, 5.0, 0.5)
weather_factor = st.sidebar.number_input("Weather Factor", 0.0, 350.0, 150.0)
channel_busy_ratio_pct = st.sidebar.number_input("Channel Busy Ratio (%)", 0.0, 100.0, 30.0)
packet_loss_pct = st.sidebar.number_input("Packet Loss (%)", 0.0, 50.0, 5.0)
speed_density_ratio = st.sidebar.number_input("Speed/Density Ratio", 0.0, 70.0, 1.7)
congestion_pressure = st.sidebar.number_input("Congestion Pressure", 0.0, 130.0, 5.0)
wireless_congestion_intensity = st.sidebar.number_input("Wireless Congestion Intensity", 0.0, 35.0, 1.6)
throughput_per_queued_vehicle = st.sidebar.number_input("Throughput/Queued Vehicle", 0.0, 22000.0, 200.0)

if st.sidebar.button("Predict Traffic Demand"):
    raw_input = {
        'timestamp': timestamp,
        'road_segment_id': road_segment_id,
        'avg_speed_kmph': avg_speed_kmph,
        'density_veh_per_km': density_veh_per_km,
        'avg_wait_time_s': avg_wait_time_s,
        'occupancy_pct': occupancy_pct,
        'flow_veh_per_hr': flow_veh_per_hr,
        'queue_length_veh': queue_length_veh,
        'incident_num': incident_num,
        'weather_factor': weather_factor,
        'channel_busy_ratio_pct': channel_busy_ratio_pct,
        'packet_loss_pct': packet_loss_pct,
        'speed_density_ratio': speed_density_ratio,
        'congestion_pressure': congestion_pressure,
        'wireless_congestion_intensity': wireless_congestion_intensity,
        'throughput_per_queued_vehicle': throughput_per_queued_vehicle,
    }

    pred_label, probs = predictor.predict_from_raw_row(raw_input)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Predicted Traffic State")
        st.markdown(f"## {pred_label}")

    with col2:
        st.subheader("Class Probabilities")
        prob_df = pd.DataFrame(list(probs.items()), columns=['Class', 'Probability'])
        st.bar_chart(prob_df.set_index('Class'))