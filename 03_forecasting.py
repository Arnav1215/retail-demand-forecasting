import pandas as pd
import numpy as np
import xgboost as xgb
import holidays

def generate_forecast(data_path, store_id, item_id, forecast_days=7):
    print(f"Loading full dataset to forecast Store {store_id}, Item {item_id}...")
    df = pd.read_csv(data_path)
    
    features = ['store', 'item', 'day', 'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'is_holiday']
    target = 'sales'
    
    X = df[features]
    y = df[target]
    
    print("Training final reproducible XGBoost model on ALL historical data...")
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    
    print(f"Generating features for the next {forecast_days} days...")
    df['date'] = pd.to_datetime(df['date'])
    last_date = df['date'].max()
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, forecast_days + 1)]
    
    future_df = pd.DataFrame({'date': future_dates})
    future_df['store'] = store_id
    future_df['item'] = item_id
    future_df['day'] = future_df['date'].dt.day
    future_df['month'] = future_df['date'].dt.month
    future_df['dayofweek'] = future_df['date'].dt.dayofweek
    
    future_df['month_sin'] = np.sin(2 * np.pi * future_df['month'] / 12)
    future_df['month_cos'] = np.cos(2 * np.pi * future_df['month'] / 12)
    future_df['dayofweek_sin'] = np.sin(2 * np.pi * future_df['dayofweek'] / 7)
    future_df['dayofweek_cos'] = np.cos(2 * np.pi * future_df['dayofweek'] / 7)
    
    years = future_df['date'].dt.year.unique()
    india_holidays = holidays.India(years=years)
    future_df['is_holiday'] = future_df['date'].apply(lambda x: 1 if x in india_holidays else 0)
    
    X_future = future_df[features]
    
    print("Predicting future demand...")
    future_df['predicted_sales'] = np.round(model.predict(X_future))
    
    output_file = f"data/forecast_store{store_id}_item{item_id}.csv"
    future_df[['date', 'store', 'item', 'predicted_sales']].to_csv(output_file, index=False)
    
    print(f"\nForecast complete! Saved to {output_file}")
    print("\n--- Upcoming 7-Day Forecast ---")
    print(future_df[['date', 'predicted_sales']])

if __name__ == "__main__":
    generate_forecast("data/processed_train.csv", store_id=1, item_id=1, forecast_days=7)