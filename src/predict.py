import pandas as pd
import numpy as np
import joblib
import json
import sys
import os

sys.path.append(os.path.dirname(__file__))
from data_preprocessing import load_and_clean_data
from feature_engineering import engineer_features, FINAL_FEATURES


class TrafficDemandPredictor:
    def __init__(self, model_dir='../models/'):
        self.lgb_model = joblib.load(f'{model_dir}lightgbm_model.pkl')
        self.xgb_model = joblib.load(f'{model_dir}xgboost_model.pkl')
        self.label_encoder = joblib.load(f'{model_dir}label_encoder.pkl')
        self.road_freq_map = joblib.load(f'{model_dir}road_freq_map.pkl')

        with open(f'{model_dir}ensemble_config.json', 'r') as f:
            config = json.load(f)
        self.w_lgb = config['weight_lgb']
        self.w_xgb = config['weight_xgb']
        self.features = config['features']

    def predict_from_raw_row(self, raw_dict):
        """
        raw_dict: dict with raw input fields (timestamp, road_segment_id, and all
        original numeric columns needed to compute FINAL_FEATURES).
        Returns: predicted_label (str), probability_dict
        """
        df = pd.DataFrame([raw_dict])
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Clip invalid negatives (same as training)
        for col in ['queue_length_veh', 'congestion_pressure', 'incident_num']:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)

        df, _ = engineer_features(df, road_freq_map=self.road_freq_map)

        X_input = df[self.features]

        lgb_proba = self.lgb_model.predict_proba(X_input)
        xgb_proba = self.xgb_model.predict_proba(X_input)
        ensemble_proba = (self.w_lgb * lgb_proba) + (self.w_xgb * xgb_proba)

        pred_class = np.argmax(ensemble_proba, axis=1)[0]
        pred_label = self.label_encoder.inverse_transform([pred_class])[0]

        prob_dict = dict(zip(self.label_encoder.classes_, ensemble_proba[0]))

        return pred_label, prob_dict