import pandas as pd
import numpy as np
import holidays
import os

def process_data(file_path, output_path):
    print("Loading raw data...")
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    print("Extracting calendar and cyclical features...")
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['dayofweek'] = df['date'].dt.dayofweek
    
    # Cyclical Month and Day
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    print("Mapping regional holidays...")
    years = df['date'].dt.year.unique()
    india_holidays = holidays.India(years=years)
    df['is_holiday'] = df['date'].apply(lambda x: 1 if x in india_holidays else 0)
    
    print("Applying IQR anomaly filtering...")
    Q1 = df['sales'].quantile(0.25)
    Q3 = df['sales'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    
    initial_shape = df.shape[0]
    df_clean = df[df['sales'] <= upper_bound]
    removed_rows = initial_shape - df_clean.shape[0]
    
    print(f"Removed {removed_rows} anomalous rows.")
    df_clean.to_csv(output_path, index=False)
    print(f"Pipeline complete. Processed data saved to {output_path}")

if __name__ == "__main__":
    input_file = "data/train.csv"
    output_file = "data/processed_train.csv"
    if os.path.exists(input_file):
        process_data(input_file, output_file)
    else:
        print(f"Error: Could not find {input_file}.")