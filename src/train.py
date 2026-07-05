import pandas as pd
import numpy as np
import joblib
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score

import sys
sys.path.append('.')
from data_preprocessing import load_and_clean_data
from feature_engineering import engineer_features, FINAL_FEATURES


def train_pipeline(data_path='../data/raw/vanet_traffic_data.csv', model_dir='../models/'):
    # 1. Load and preprocess
    df = load_and_clean_data(data_path)
    df, road_freq_map = engineer_features(df)

    # 2. Encode label
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])

    X = df[FINAL_FEATURES]
    y = df['label_encoded']

    # 3. Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Train tuned LightGBM (best params from Optuna Step 7)
    lgb_model = lgb.LGBMClassifier(
        n_estimators=244, learning_rate=0.0134, num_leaves=33, max_depth=9,
        min_child_samples=81, subsample=0.7678, colsample_bytree=0.8745,
        objective='multiclass', num_class=4, random_state=42, n_jobs=-1, verbosity=-1
    )
    lgb_model.fit(X_train, y_train)

    # 5. Train tuned XGBoost (best params from Optuna Step 7)
    xgb_model = xgb.XGBClassifier(
        n_estimators=234, learning_rate=0.10698, max_depth=4, min_child_weight=4,
        subsample=0.6826, colsample_bytree=0.6687,
        objective='multi:softmax', num_class=4, random_state=42, n_jobs=-1, eval_metric='mlogloss'
    )
    xgb_model.fit(X_train, y_train)

    # 6. Evaluate
    lgb_preds = lgb_model.predict(X_test)
    xgb_preds = xgb_model.predict(X_test)
    print("LightGBM Test Accuracy:", accuracy_score(y_test, lgb_preds))
    print("XGBoost Test Accuracy:", accuracy_score(y_test, xgb_preds))

    # 7. Save artifacts
    joblib.dump(lgb_model, f'{model_dir}lightgbm_model.pkl')
    joblib.dump(xgb_model, f'{model_dir}xgboost_model.pkl')
    joblib.dump(le, f'{model_dir}label_encoder.pkl')
    joblib.dump(road_freq_map, f'{model_dir}road_freq_map.pkl')

    with open(f'{model_dir}ensemble_config.json', 'w') as f:
        json.dump({'weight_lgb': 0.5, 'weight_xgb': 0.5, 'features': FINAL_FEATURES}, f, indent=4)

    print("\nAll artifacts saved to", model_dir)
    return lgb_model, xgb_model, le


if __name__ == '__main__':
    train_pipeline()