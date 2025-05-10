import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib

tscv = TimeSeriesSplit(n_splits=3)

# Đường dẫn đến file dữ liệu
data_path = Path("D:/PROJECT_II/freqtrade/user_data/data/binance/BTC_USDT-1h.feather")
df = pd.read_feather(data_path)

print("Dữ liệu:")
print(df.head())  # In 5 dòng đầu của dữ liệu

# Chuyển cột 'date' thành kiểu datetime
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Lọc dữ liệu trong khoảng thời gian từ
start_date = '2023-11-01'
end_date = '2024-06-01'

df = df[(df.index >= start_date) & (df.index <= end_date)]

# In ra dữ liệu sau khi lọc
print("\nDữ liệu sau khi lọc:")
print(df.head())  # In ra 5 dòng đầu tiên của dữ liệu đã lọc
print(df.tail())

# Tạo biến mục tiêu: giá đóng cửa của giờ tiếp theo
df['target'] = df['close'].shift(-1)

# Feature Engineering - tạo các đặc trưng kỹ thuật
df['ma_3_open'] = df['open'].rolling(window=3).mean()
df['ma_3_high'] = df['high'].rolling(window=3).mean()
df['ma_3_low'] = df['low'].rolling(window=3).mean()
df['ma_3_close'] = df['close'].rolling(window=3).mean()

df['ma_6_open'] = df['open'].rolling(window=6).mean()
df['ma_6_high'] = df['high'].rolling(window=6).mean()
df['ma_6_low'] = df['low'].rolling(window=6).mean()
df['ma_6_close'] = df['close'].rolling(window=6).mean()

df['std_3_open'] = df['open'].rolling(window=3).std()
df['std_3_high'] = df['high'].rolling(window=3).std()
df['std_3_low'] = df['low'].rolling(window=3).std()
df['std_3_close'] = df['close'].rolling(window=3).std()

# EMA
df['ema_10_open'] = df['open'].ewm(span=10, adjust=False).mean()
df['ema_10_high'] = df['high'].ewm(span=10, adjust=False).mean()
df['ema_10_low'] = df['low'].ewm(span=10, adjust=False).mean()
df['ema_10_close'] = df['close'].ewm(span=10, adjust=False).mean()

# Tính RSI cho close
delta_close = df['close'].diff()
gain_close = (delta_close.where(delta_close > 0, 0)).rolling(14).mean()
loss_close = (-delta_close.where(delta_close < 0, 0)).rolling(14).mean()
rs_close = gain_close / loss_close
df['rsi_14_close'] = 100 - (100 / (1 + rs_close))

# Tính RSI cho open
delta_open = df['open'].diff()
gain_open = (delta_open.where(delta_open > 0, 0)).rolling(14).mean()
loss_open = (-delta_open.where(delta_open < 0, 0)).rolling(14).mean()
rs_open = gain_open / loss_open
df['rsi_14_open'] = 100 - (100 / (1 + rs_open))

# Tính RSI cho high
delta_high = df['high'].diff()
gain_high = (delta_high.where(delta_high > 0, 0)).rolling(14).mean()
loss_high = (-delta_high.where(delta_high < 0, 0)).rolling(14).mean()
rs_high = gain_high / loss_high
df['rsi_14_high'] = 100 - (100 / (1 + rs_high))

# Tính RSI cho low
delta_low = df['low'].diff()
gain_low = (delta_low.where(delta_low > 0, 0)).rolling(14).mean()
loss_low = (-delta_low.where(delta_low < 0, 0)).rolling(14).mean()
rs_low = gain_low / loss_low
df['rsi_14_low'] = 100 - (100 / (1 + rs_low))


# MACD
ema_12_open = df['open'].ewm(span=12, adjust=False).mean()
ema_26_open = df['open'].ewm(span=26, adjust=False).mean()
df['macd_open'] = ema_12_open - ema_26_open

ema_12_high = df['high'].ewm(span=12, adjust=False).mean()
ema_26_high = df['high'].ewm(span=26, adjust=False).mean()
df['macd_high'] = ema_12_high - ema_26_high

ema_12_low = df['low'].ewm(span=12, adjust=False).mean()
ema_26_low = df['low'].ewm(span=26, adjust=False).mean()
df['macd_low'] = ema_12_low - ema_26_low

ema_12_close = df['close'].ewm(span=12, adjust=False).mean()
ema_26_close = df['close'].ewm(span=26, adjust=False).mean()
df['macd_close'] = ema_12_close - ema_26_close

# Loại bỏ các dòng có giá trị NaN
df.dropna(inplace=True)

# Chọn các đặc trưng đầu vào (features) và đầu ra (target)
features = df[['open', 'high', 'low', 'close', 'volume', 
               'ma_3_open', 'ma_3_high', 'ma_3_low', 'ma_3_close',
               'ma_6_open', 'ma_6_high', 'ma_6_low', 'ma_6_close',
               'std_3_open', 'std_3_high', 'std_3_low', 'std_3_close',
               'ema_10_open', 'ema_10_high', 'ema_10_low', 'ema_10_close',
               'rsi_14_open', 'rsi_14_high', 'rsi_14_low',
               'macd_close', 'macd_open', 'macd_high', 'macd_low']]
target = df['target']

# Chia dữ liệu thành train và test
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, shuffle=False
)

# Kiểm tra kết quả
print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

# Huấn luyện mô hình Random Forest
param_grid = {
    'n_estimators': [100, 200, 400, 600],
    'max_depth': [5, 10, 100],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestRegressor(random_state=42)
grid_search_rf = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)
# tuning hyperparameters
grid_search_rf.fit(X_train, y_train)
best_rf_model = grid_search_rf.best_estimator_

# Dự đoán và đánh giá cho Random Forest
y_pred_rf = best_rf_model.predict(X_test)

rf_predictions = pd.DataFrame({'Date': X_test.index, 'Prediction': y_pred_rf})
print("Dự đoán Random Forest:")
print(rf_predictions)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

# Huấn luyện mô hình XGBoost
xgb_model = xgb.XGBRegressor(random_state=42)

# Grid Search để tìm hyperparameters tốt nhất cho XGBoost
param_grid_xgb = {
    'n_estimators': [50, 150],
    'max_depth': [5, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0]
}

grid_search_xgb = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid_xgb,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)

# tuning hyperparameters
grid_search_xgb.fit(X_train, y_train)
best_xgb_model = grid_search_xgb.best_estimator_

# Dự đoán và đánh giá cho XGBoost
y_pred_xgb = best_xgb_model.predict(X_test)

xgb_predictions = pd.DataFrame({'Date': X_test.index, 'Prediction': y_pred_xgb})
print("Dự đoán XGBoost:")
print(xgb_predictions)

mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

# KNN Regressor
scaler_knn = StandardScaler()
X_train_knn = scaler_knn.fit_transform(X_train)
X_test_knn = scaler_knn.transform(X_test)

knn = KNeighborsRegressor()
param_grid_knn = {
    'n_neighbors': [1, 50]
}
grid_search_knn = GridSearchCV(
    estimator=knn,
    param_grid=param_grid_knn,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)
grid_search_knn.fit(X_train_knn, y_train)
best_knn_model = grid_search_knn.best_estimator_

y_pred_knn = best_knn_model.predict(X_test_knn)

knn_predictions = pd.DataFrame({'Date': X_test.index, 'Prediction': y_pred_knn})
print("Dự đoán KNN:")
print(knn_predictions)

mae_knn = mean_absolute_error(y_test, y_pred_knn)
rmse_knn = np.sqrt(mean_squared_error(y_test, y_pred_knn))
r2_knn = r2_score(y_test, y_pred_knn)

# Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=42)
param_grid_dt = {
    'max_depth': [5, 10, 50],
    'min_samples_split': [2, 5, 10]
}
grid_search_dt = GridSearchCV(
    estimator=dt,
    param_grid=param_grid_dt,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)
grid_search_dt.fit(X_train, y_train)
best_dt_model = grid_search_dt.best_estimator_

y_pred_dt = best_dt_model.predict(X_test)

dt_predictions = pd.DataFrame({'Date': X_test.index, 'Prediction': y_pred_dt})
print("Dự đoán Decision Tree:")
print(dt_predictions)

mae_dt = mean_absolute_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
r2_dt = r2_score(y_test, y_pred_dt)


# SVR Regressor
scaler_svr = StandardScaler()
X_train_scaled = scaler_svr.fit_transform(X_train)
X_test_scaled = scaler_svr.transform(X_test)

# Huấn luyện
svr = SVR()
param_grid_svr = {
    'kernel': ['rbf'],
    'C': [1, 10],
    'epsilon': [0.1, 0.2]
}
grid_search_svr = GridSearchCV(
    estimator=svr,
    param_grid=param_grid_svr,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)
grid_search_svr.fit(X_train_scaled, y_train)
best_svr_model = grid_search_svr.best_estimator_

# Dự đoán và đánh giá cho SVR
y_pred_svr = best_svr_model.predict(X_test_scaled)

svr_predictions = pd.DataFrame({'Date': X_test.index, 'Prediction': y_pred_svr})
print("Dự đoán SVR:")
print(svr_predictions)

mae_svr = mean_absolute_error(y_test, y_pred_svr)
rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))
r2_svr = r2_score(y_test, y_pred_svr)

print("Best parameters for Random Forest:", grid_search_rf.best_params_)
print("Best parameters for XGBoost:", grid_search_xgb.best_params_)
print("Best parameters for KNN:", grid_search_knn.best_params_)
print("Best parameters for Decision Tree:", grid_search_dt.best_params_)
print("Best parameters for SVR:", grid_search_svr.best_params_)

# kết quả đánh giá dựa trên các chỉ số
results = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost', 'KNN', 'Decision Tree', 'SVR'],
    'MAE': [mae_rf, mae_xgb, mae_knn, mae_dt, mae_svr],
    'RMSE': [rmse_rf, rmse_xgb, rmse_knn, rmse_dt, rmse_svr],
    'R²': [r2_rf, r2_xgb, r2_knn, r2_dt, r2_svr]
})
print(results)


# Dữ liệu các chỉ số
metrics = ['MAE', 'RMSE', 'R²']
scores = {
    'MAE': [mae_rf, mae_xgb, mae_knn, mae_dt, mae_svr],
    'RMSE': [rmse_rf, rmse_xgb, rmse_knn, rmse_dt, rmse_svr],
    'R²': [r2_rf, r2_xgb, r2_knn, r2_dt, r2_svr]
}
models = ['Random Forest', 'XGBoost', 'KNN', 'Decision Tree', 'SVR']
colors = ['lightblue', 'pink', 'lightgreen', 'orange', 'violet']

results_df = results

# Vẽ 3 biểu đồ
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
metrics = ['MAE', 'RMSE', 'R²']

for i, metric in enumerate(metrics):
    ax = axes[i]
    values = scores[metric]
    bars = ax.bar(models, values, color=colors)

    ax.set_title(metric)
    ax.set_ylim(0, max(values) * 1.15)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height * 0.98),
                    xytext=(0, -10),
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.tight_layout()
plt.show()

y_pred_rf       # Dự đoán từ Random Forest  
y_pred_xgb      # Dự đoán từ XGBoost  
y_pred_knn      # Dự đoán từ KNN  
y_pred_dt       # Dự đoán từ Decision Tree  
y_pred_svr      # Dự đoán từ SVR  


# Vẽ dự đoán 100h sau của 5 mô hình
plt.figure(figsize=(15, 6))
plt.plot(y_test[-100:].values, label='Thực tế', color='black', linewidth=2)

plt.plot(y_pred_rf[-100:], label='Random Forest', linestyle='--')
plt.plot(y_pred_xgb[-100:], label='XGBoost', linestyle='--')
plt.plot(y_pred_knn[-100:], label='KNN', linestyle='--')
plt.plot(y_pred_dt[-100:], label='Decision Tree', linestyle='--')
plt.plot(y_pred_svr[-100:], label='SVR', linestyle='--')

plt.title('So sánh dự đoán của các mô hình (100 giờ cuối)', fontsize=14)
plt.xlabel('Thời gian (index)', fontsize=12)
plt.ylabel('Giá BTC-USDT', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Bieu do so sanh Gia Thực te với RF
plt.figure(figsize=(13, 4))
plt.plot(y_test[-100:].values, label='Thực tế', color='black', linewidth=2)
plt.plot(y_pred_rf[-100:], label='Random Forest', linestyle='--')
plt.title('So sánh dự đoán của mô hình Random Forest với Giá Thực tế (100 giờ cuối)', fontsize=13)
plt.xlabel('Thời gian (index)', fontsize=12)
plt.ylabel('Giá BTC-USDT', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Lưu mô hình vào file
joblib.dump(best_rf_model, 'best_rf_model.pkl')
