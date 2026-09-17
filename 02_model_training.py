import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt

def train_and_evaluate(data_path):
    print("Loading processed data...")
    df = pd.read_csv(data_path)
    df = df.sort_values('date')
    
    # Updated feature list dropping raw dayofweek and keeping the cyclical versions
    features = ['store', 'item', 'day', 'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'is_holiday']
    target = 'sales'
    
    print("Performing time-based split (Train: 2013-2016, Test: 2017)...")
    train_df = df[df['date'] < '2017-01-01']
    test_df = df[df['date'] >= '2017-01-01']
    
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]
    
    # Added random_state=42 to XGBoost for reproducibility
    models = {
        "OLS Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "XGBoost Regressor": xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    }
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, train_predictions)
        test_mae = mean_absolute_error(y_test, test_predictions)
        
        print(f">>> {name} Train MAE: {train_mae:.2f}")
        print(f">>> {name} Test MAE:  {test_mae:.2f}")
        
        if name == "XGBoost Regressor":
            # 1. Save Feature Importance Plot
            xgb.plot_importance(model, max_num_features=10, importance_type='weight')
            plt.title("XGBoost Feature Importance")
            plt.tight_layout()
            plt.savefig("data/xgboost_feature_importance.png")
            print(">>> Feature importance plot saved to data/xgboost_feature_importance.png")
            plt.close() # Clear the canvas
            
            # 2. Save Backtest Plot safely (Bulletproof Pandas indexing)
            plot_df = test_df.copy()
            plot_df['actual'] = y_test
            plot_df['predicted'] = test_predictions
            
            mask = (plot_df['store'] == 1) & (plot_df['item'] == 1)
            subset = plot_df[mask].head(30) # Grab the first 30 days
            
            plt.figure(figsize=(12, 5))
            plt.plot(subset['date'], subset['actual'], label='Actual Sales', marker='o')
            plt.plot(subset['date'], subset['predicted'], label='XGBoost Predicted', marker='x', linestyle='--')
            plt.title("Backtest: Actual vs Predicted Sales (Store 1, Item 1 - Jan 2017)")
            plt.ylabel("Sales")
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig("data/backtest_predictions.png")
            print(">>> Backtest plot saved to data/backtest_predictions.png")
            plt.close()

if __name__ == "__main__":
    train_and_evaluate("data/processed_train.csv")