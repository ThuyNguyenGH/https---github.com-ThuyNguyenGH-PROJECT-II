import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import joblib

# Đường dẫn đến file dữ liệu
data_path = Path("D:/PROJECT_II/freqtrade/user_data/data/binance/BTC_USDT-1h.feather")
df = pd.read_feather(data_path)

print("Dữ liệu:")
print(df.head())  # In 5 dòng đầu của dữ liệu

# Chuyển cột 'date' thành kiểu datetime
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Lọc dữ liệu trong khoảng thời gian từ 1/9/2023 đến 1/5/2024
start_date = '2023-09-01'
end_date = '2024-05-01'

df = df[(df.index >= start_date) & (df.index <= end_date)]

# In ra dữ liệu sau khi lọc
print("\nDữ liệu sau khi lọc:")
print(df.head())  # In ra 5 dòng đầu tiên của dữ liệu đã lọc
print(df.tail())

# Tạo biến mục tiêu: giá đóng cửa của giờ tiếp theo
df['target'] = df['close'].shift(-1)

# Feature Engineering - tạo các đặc trưng kỹ thuật
df['ma_3'] = df['close'].rolling(window=3).mean()    # Trung bình 3 giờ
df['ma_6'] = df['close'].rolling(window=6).mean()    # Trung bình 6 giờ
df['std_3'] = df['close'].rolling(window=3).std()    # Độ lệch chuẩn 3 giờ

# EMA
df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()

# RSI
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df['rsi_14'] = 100 - (100 / (1 + rs))

# MACD
ema_12 = df['close'].ewm(span=12, adjust=False).mean()
ema_26 = df['close'].ewm(span=26, adjust=False).mean()
df['macd'] = ema_12 - ema_26

# Loại bỏ các dòng có giá trị NaN
df.dropna(inplace=True)

# Chọn các đặc trưng đầu vào (features) và đầu ra (target)
features = df[['close', 'volume', 'ma_3', 'ma_6', 'std_3', 'ema_10', 'rsi_14', 'macd']]
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
    'max_depth': [10, 30, 50, 100, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# RandomizedSearchCV
rf = RandomForestRegressor(random_state=42)
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=50,  # Số lần thử nghiệm
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

# 3. Huấn luyện mô hình
random_search.fit(X_train, y_train)
best_rf_model = random_search.best_estimator_

# 4. Đánh giá bằng Cross-validation trên tập train
cv_mae_scores = cross_val_score(best_rf_model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
print("Cross-validated MAE (Train):", -np.mean(cv_mae_scores))

# 5. Dự đoán trên tập test
y_pred_rf = best_rf_model.predict(X_test)

# 6. Đánh giá hiệu năng
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
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)

# tuning hyperparameters
grid_search_xgb.fit(X_train, y_train)
best_xgb_model = grid_search_xgb.best_estimator_

# Dự đoán và đánh giá cho XGBoost
y_pred_xgb = best_xgb_model.predict(X_test)

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
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)
grid_search_knn.fit(X_train_knn, y_train)
best_knn_model = grid_search_knn.best_estimator_

y_pred_knn = best_knn_model.predict(X_test_knn)
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
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)
grid_search_dt.fit(X_train, y_train)
best_dt_model = grid_search_dt.best_estimator_

y_pred_dt = best_dt_model.predict(X_test)
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
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)
grid_search_svr.fit(X_train_scaled, y_train)
best_svr_model = grid_search_svr.best_estimator_

# Dự đoán và đánh giá cho SVR
y_pred_svr = best_svr_model.predict(X_test_scaled)
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


# Vẽ dự đoán 100h đầu tiên của 5 mô hình
plt.figure(figsize=(15, 6))
plt.plot(y_test[:100].values, label='Thực tế', color='black', linewidth=2)

plt.plot(y_pred_rf[:100], label='Random Forest', linestyle='--')
plt.plot(y_pred_xgb[:100], label='XGBoost', linestyle='--')
plt.plot(y_pred_knn[:100], label='KNN', linestyle='--')
plt.plot(y_pred_dt[:100], label='Decision Tree', linestyle='--')
plt.plot(y_pred_svr[:100], label='SVR', linestyle='--')

plt.title('So sánh dự đoán của các mô hình (100 giờ đầu tiên)', fontsize=14)
plt.xlabel('Thời gian (index)', fontsize=12)
plt.ylabel('Giá BTC-USDT', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# Bieu do so sanh Gia Thực te với RF
plt.figure(figsize=(13, 4))
plt.plot(y_test[:100].values, label='Thực tế', color='black', linewidth=2)
plt.plot(y_pred_rf[:100], label='Random Forest', linestyle='--')
plt.title('So sánh dự đoán của mô hình Random Forest với Giá Thực tế (100 giờ đầu tiên)', fontsize=13)
plt.xlabel('Thời gian (index)', fontsize=12)
plt.ylabel('Giá BTC-USDT', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Lưu mô hình vào file
joblib.dump(best_rf_model, 'random_forest_model.pkl')
