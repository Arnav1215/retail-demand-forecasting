import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading data for EDA...")
df = pd.read_csv("data/train.csv")
df['date'] = pd.to_datetime(df['date'])
df['dayofweek'] = df['date'].dt.dayofweek

print("Generating plots...")
plt.figure(figsize=(15, 5))

#1.Sales Trend over time
plt.subplot(1, 3, 1)
df.set_index('date')['sales'].resample('ME').mean().plot()
plt.title("Average Monthly Sales Trend")
plt.ylabel("Sales")

#2. Sales by Day of Week
plt.subplot(1, 3, 2)
sns.barplot(x='dayofweek', y='sales', data=df)
plt.title("Avg Sales by Day of Week")
plt.xlabel("Day (0=Mon, 6=Sun)")

#3. Store Sales Distribution
plt.subplot(1, 3, 3)
sns.histplot(df['sales'], bins=50, kde=True)
plt.title("Distribution of Sales")

plt.tight_layout()
plt.savefig("data/eda_visualizations.png")
print("EDA plots saved to data/eda_visualizations.png")