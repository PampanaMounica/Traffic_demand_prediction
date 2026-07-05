# 🚦 Traffic Demand Prediction using LightGBM & XGBoost

A machine learning system that predicts real-time traffic congestion levels
(Free-flow, Moderate, Heavy, Gridlock) using VANET (Vehicular Ad-hoc Network)
traffic, weather, and wireless communication data. Built as an internship
project demonstrating end-to-end ML pipeline development, from raw data to
deployed web application.

## 🎯 Problem Statement
Multi-class classification: predict traffic congestion state from 12 traffic,
weather, and wireless network features, using an ensemble of LightGBM and
XGBoost classifiers.

## 📊 Dataset
- **Source:** Synthetic VANET traffic dataset (195,714 records, 500 road segments)
- **Target:** 4-class congestion label (Free-flow / Moderate / Heavy / Gridlock)
- Full documentation: [`data/README.md`](data/README.md)

## 🧠 Methodology
1. **EDA** — distribution analysis, correlation heatmap, class balance check
2. **Multicollinearity analysis** — VIF scoring + Random Forest feature importance
3. **Feature selection** — reduced 26 raw features to 17 evidence-based features
   (data-driven, validated via A/B testing with LightGBM/XGBoost, not assumption-based)
4. **Feature engineering** — time-based features (hour, rush-hour flag, weekend flag),
   frequency encoding for road segments
5. **Model training** — LightGBM and XGBoost, tuned via Optuna (30 trials, 5-fold CV)
6. **Ensemble** — weighted probability combination of both tuned models
7. **Leakage verification** — decision-stump and ablation testing confirmed no
   target leakage; high accuracy attributed to well-separated synthetic classes
8. **Deployment** — interactive Streamlit web application

## 📈 Results

| Model | Accuracy | F1 (macro) |
|---|---|---|
| LightGBM (tuned) | 99.06% | 0.9906 |
| XGBoost (tuned) | 99.06% | 0.9906 |
| Ensemble (50/50) | 99.06% | 0.9906 |

**Note:** This dataset is synthetic, with classes generated via threshold rules
on the underlying features. This produces cleaner class separation than
real-world sensor data would show. See [project report](reports/project_report.md)
for full leakage analysis and discussion of this limitation.

## 🗂️ Project Structure

traffic-demand-prediction/
├── data/
│   ├── raw/vanet_traffic_data.csv
│   └── README.md
├── notebooks/
│   └── traffic_demand_prediction.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── predict.py
├── models/
│   ├── lightgbm_model.pkl
│   ├── xgboost_model.pkl
│   ├── label_encoder.pkl
│   ├── road_freq_map.pkl
│   └── ensemble_config.json
├── app/
│   └── streamlit_app.py
├── reports/
│   ├── figures/
│   └── project_report.md
├── requirements.txt
└── README.md


## 🛠️ Tech Stack
Python, Pandas, NumPy, Scikit-learn, LightGBM, XGBoost, Optuna, Matplotlib,
Seaborn, Streamlit

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd traffic-demand-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (optional - pre-trained models included in /models)
cd src
python train.py

# 4. Launch the Streamlit app
cd ../app
streamlit run streamlit_app.py
```

## 🔍 Key Learnings
- Rigorous multicollinearity analysis (VIF + feature importance) prevents
  naive feature dropping based on correlation alone
- A/B testing feature removal against actual model performance provides
  stronger evidence than statistical heuristics alone
- High accuracy must be scrutinized for leakage — decision-stump and
  ablation tests are simple, effective diagnostic tools
- Ensembling doesn't always improve performance; testing multiple weight
  combinations empirically avoids blind assumption

## 👤 Author
P.Mounica — Internship Project, Innovexa Catalyst