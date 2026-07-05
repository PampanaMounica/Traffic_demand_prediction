# Project Report: Traffic Demand Prediction using LightGBM & XGBoost

## 1. Objective
Build a multi-class classification system to predict traffic congestion state
(Free-flow, Moderate, Heavy, Gridlock) from VANET traffic, weather, and wireless
communication data, using LightGBM and XGBoost with ensemble learning.

## 2. Dataset Overview
- **File:** `vanet_traffic_data.csv`
- **Size:** 195,714 rows × 27 columns
- **Road segments:** 500 unique segments
- **Time range:** 2025-09-28 to 2025-09-30 (~2.5 days, high-frequency sampling)
- **Missing values:** 0
- **Duplicate rows:** 0
- **Target distribution:** Free-flow 29.5%, Moderate 27.6%, Heavy 24.2%, Gridlock 18.6%
  (mild imbalance, addressed via stratified splitting)

Full column-level documentation: see `data/README.md`.

## 3. Exploratory Data Analysis
- Statistical summary showed several physically implausible small negative
  values (`avg_wait_time_s`, `queue_length_veh`, `incident_num`,
  `congestion_pressure`) — consistent with synthetic data generation noise,
  clipped to 0 during preprocessing.
- Correlation heatmap revealed extremely high inter-feature correlation
  across nearly all 26 features (many pairs > 0.95), atypical for real-world
  sensor data, indicating synthetic generation from shared underlying formulas.

## 4. Multicollinearity Analysis
Three diagnostics were combined to make evidence-based feature decisions:

1. **Correlation with target:** Max single-feature correlation with label was
   0.672 (`speed_density_ratio`) — ruled out simple 1:1 leakage.
2. **Variance Inflation Factor (VIF):** Several features exceeded VIF > 1000
   (e.g., `rssi_dbm` VIF 3859, `weather_factor` VIF 3443, `temp_c` VIF 2928),
   confirming severe multicollinearity.
3. **Random Forest feature importance:** Used as a cross-check — features with
   both high VIF and near-zero importance (e.g., `rssi_dbm`, `msg_rate_hz`,
   `temp_c`, importance ≈ 0.001) were confirmed safe to drop.

### Feature Selection Outcome
Reduced from 26 raw features to **17 final features**, retaining
domain-important variables (`flow_veh_per_hr`, `queue_length_veh`) per
project stakeholder input even where statistically redundant, since they
carry interpretability value for a traffic-demand use case.

Two borderline features (`signal_state_num`, `heading_deg`) were validated
via a direct **A/B test**: models trained with and without these features
produced identical accuracy (0.99062412 vs 0.99062412), providing direct
empirical evidence — not just statistical inference — that they added no
predictive value. Both were dropped on this basis.

## 5. Feature Engineering
- Extracted `hour`, `day_of_week`, `is_weekend`, `is_rush_hour` from `timestamp`
- Frequency-encoded `road_segment_id` (500 categories) to avoid high-cardinality
  one-hot expansion
- Note: dataset spans only 3 calendar days, limiting `day_of_week` variation;
  retained regardless for completeness of time-based feature engineering

## 6. Model Training & Tuning
- **Split:** 80/20 stratified train/test split (preserves class proportions)
- **Tuning:** Optuna, 30 trials per model, 5-fold stratified cross-validation,
  optimizing macro F1-score
- **Best LightGBM params:** n_estimators=244, learning_rate=0.0134, num_leaves=33,
  max_depth=9, min_child_samples=81, subsample=0.768, colsample_bytree=0.875
- **Best XGBoost params:** n_estimators=234, learning_rate=0.107, max_depth=4,
  min_child_weight=4, subsample=0.683, colsample_bytree=0.669

## 7. Results

| Model | Test Accuracy | F1 (macro) |
|---|---|---|
| LightGBM (tuned) | 0.9906 | 0.9906 |
| XGBoost (tuned) | 0.9906 | 0.9906 |
| Ensemble (50/50 weighted) | 0.9906 | 0.9906 |

Confusion matrices for both models show errors concentrated between adjacent
congestion states (Heavy↔Gridlock, Heavy↔Moderate), consistent with congestion
being a continuum rather than strictly discrete categories.

Multiple ensemble weight combinations (50/50, 55/45, 60/40, 45/55) were tested;
none improved on individual model performance, indicating both base models
already operate near the practical accuracy ceiling for this dataset. This
result is reported transparently rather than selectively omitted.

## 8. Data Leakage Verification
Given the unusually high accuracy (99%+), a dedicated leakage investigation
was conducted:

1. **Per-class feature range analysis:** Showed minimal overlap between
   classes for top features (e.g., `avg_speed_kmph`: Free-flow 39–78 km/h vs.
   Gridlock 1–30 km/h) — indicating clean class separation rather than a
   hidden shortcut feature.
2. **Single-feature decision stump test:** Nearly every retained traffic/network
   feature alone achieved ~99% accuracy as a depth-3 decision tree, while
   time/identity features (`hour`, `road_segment_freq`) performed poorly (30–70%)
   alone — confirming genuine signal distributed across features, not a single
   leaking column.
3. **Ablation test:** Removing the single most important feature
   (`congestion_pressure`) caused zero accuracy drop, showing no individual
   feature is solely responsible for model performance.

**Conclusion:** No data leakage was found. The high accuracy is attributable to
the dataset's synthetic generation process, where `label` was likely assigned
via threshold rules over the feature space, producing cleanly separated
classes atypical of noisy real-world sensor data. This distinction — between
"well-separated synthetic classes" and "data leakage" — is an important
limitation to disclose: real-world deployment on live sensor data would likely
show lower accuracy due to measurement noise and less distinct class boundaries.

## 9. Deployment
An interactive Streamlit web application (`app/streamlit_app.py`) was built,
accepting raw traffic parameter inputs and returning the predicted congestion
class with probability breakdown, using the saved ensemble pipeline.

## 10. Limitations & Future Work
- Synthetic dataset limits real-world generalization claims
- Only 2.5 days of data — no seasonal/weekly pattern validation possible
- Future work: validate on real sensor data if available; explore accident
  risk prediction and route recommendation as separate extensions