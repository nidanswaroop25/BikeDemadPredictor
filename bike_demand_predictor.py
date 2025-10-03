import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('london_merged.csv')

# Convert timestamp to datetime and create new features
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df['year'] = df['timestamp'].dt.year

# --- EDA ---
sns.set_style("whitegrid")

plt.figure(figsize=(10,6))
sns.histplot(df['cnt'], bins=50, kde=True)
plt.title('Distribution of Bike Shares')
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(x='hour', y='cnt', data=df)
plt.title('Bike Shares by Hour')
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x='day_of_week', y='cnt', data=df)
plt.title('Bike Shares by Day of Week (0=Mon, 6=Sun)')
plt.show()

plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# --- Model Building ---
features = ['t1', 't2', 'hum', 'wind_speed', 'weather_code',
            'is_holiday', 'is_weekend', 'season',
            'hour', 'day_of_week', 'month', 'year']

X = df[features]
y = df['cnt']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# --- Evaluation ---
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Feature Importance
fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=fi, y=fi.index)
plt.title('Feature Importances')
plt.show()

# Actual vs Predicted
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.title('Actual vs Predicted Bike Shares')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()
