# Retail Demand Forecasting Pipeline

An end-to-end machine learning pipeline for cleaning, feature-engineering, modeling, and forecasting multi-store retail sales data, built as reproducible, modular Python scripts.

## 🚀 Project Overview
Accurate demand forecasting is critical for optimizing inventory, minimizing waste, and preventing stockouts. This project implements a modular, script-driven pipeline that replaces messy Jupyter notebooks with a clean, organized architecture — comparing linear baselines (OLS, Ridge) against a non-linear gradient boosting model (XGBoost) for daily sales prediction across multiple stores and items.

## 🛠️ Project Structure
```text
retail-demand-forecasting/
│
├── data/                              # Raw, processed data and visual outputs
│   ├── train.csv
│   ├── processed_train.csv
│   ├── forecast_store1_item1.csv
│   ├── eda_visualizations.png
│   ├── xgboost_feature_importance.png
│   └── backtest_predictions.png
│
├── 00_eda.py                          # Exploratory data analysis & visual trend generation
├── 01_data_processing.py              # Calendar extraction, cyclical encoding, holiday mapping, & IQR anomaly filtering
├── 02_model_training.py               # Time-based split, OLS/Ridge baselines, XGBoost training, backtest & feature importance plots
├── 03_forecasting.py                  # Final model retraining on full history and 7-day future demand generation
├── requirements.txt                   # Pinned project dependencies
└── README.md
```

## ⚙️ Pipeline Steps
1. **`00_eda.py`** — Loads raw data and generates exploratory visualizations (sales distribution, trends by day/weekend/holiday, correlation heatmap).
2. **`01_data_processing.py`** — Extracts calendar features (day, month, day-of-week), applies cyclical (sine/cosine) encoding to month and day-of-week, maps Indian public holidays, and removes anomalous sales values using IQR-based filtering.
3. **`02_model_training.py`** — Performs a **time-based train/test split** (train: 2013–2016, test: 2017) to avoid data leakage, trains OLS, Ridge, and XGBoost models, reports train/test MAE for each, and saves a feature importance plot and a backtest plot (actual vs. predicted sales for a sample store/item).
4. **`03_forecasting.py`** — Retrains XGBoost on the full historical dataset and generates a 7-day forward sales forecast for a given store/item.

## 📊 Key Results
| Model              | Train MAE | Test MAE |
|--------------------|-----------|----------|
| OLS Regression     | 20.30     | 22.32    |
| Ridge Regression   | 20.30     | 22.32    |
| XGBoost Regressor  | 8.10      | 10.00    |

XGBoost reduced test MAE by over **50%** compared to the linear baselines, driven by its ability to capture non-linear interactions between store, item, and calendar features.

*(OLS and Ridge produce nearly identical results here — with `alpha=1.0`, the L2 penalty has minimal effect given the feature set, so Ridge doesn't diverge meaningfully from plain OLS.)*

## 📈 Backtest
`data/backtest_predictions.png` compares actual vs. predicted daily sales for a sample store/item over a 30-day window. The model captures the broad weekly/seasonal pattern but underfits sharper day-to-day spikes — expected, since the current features are purely calendar-based with no memory of recent actual sales (see Future Improvements below).

## 🔭 Future Improvements
- Add lag and rolling-window features (e.g., sales from 1, 7, 14 days ago) to capture short-term demand momentum
- Use `TimeSeriesSplit` for more robust cross-validation across multiple time windows, rather than a single train/test split
- Use XGBoost's native categorical handling (`enable_categorical=True`) for `store` and `item` instead of raw integer encoding
- Hyperparameter tuning for XGBoost (currently using reasonable defaults)

## 🧰 Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python3 00_eda.py
python3 01_data_processing.py
python3 02_model_training.py
python3 03_forecasting.py
```

## 🧱 Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, `holidays`
