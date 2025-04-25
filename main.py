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

# Đường dẫn đến file dữ liệu
data_path = Path("D:/PROJECT_II/PROJECT-II/freqtrade/user_data/data/binance/BTC_USDT-1h.feather")
df = pd.read_feather(data_path)

# Chuyển cột 'date' thành kiểu datetime và đặt làm chỉ mục
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Tạo biến mục tiêu: giá đóng cửa của giờ tiếp theo
df['target'] = df['close'].shift(-1)

# Feature Engineering - tạo các đặc trưng kỹ thuật
df['ma_3'] = df['close'].rolling(window=3).mean()    # Trung bình 3 giờ
df['ma_6'] = df['close'].rolling(window=6).mean()    # Trung bình 6 giờ
df['std_3'] = df['close'].rolling(window=3).std()    # Độ lệch chuẩn 3 giờ

# Loại bỏ các dòng có giá trị NaN
df.dropna(inplace=True)

# Chọn các đặc trưng đầu vào (features) và đầu ra (target)
features = df[['close', 'volume', 'ma_3', 'ma_6', 'std_3']]
target = df['target']

# Chia dữ liệu thành train và test
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, shuffle=False
)


# Huấn luyện mô hình Random Forest
# Grid Search để tìm hyperparameters tốt nhất
param_grid = {
    'n_estimators': [300, 600],
    'max_depth': [5, 10, 100],
    'min_samples_split': [2, 5]
}

rf = RandomForestRegressor(random_state=42)
grid_search_rf = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)

# Huấn luyện mô hình với tuning hyperparameters
grid_search_rf.fit(X_train, y_train)
best_rf_model = grid_search_rf.best_estimator_

# Dự đoán và đánh giá cho Random Forest
y_pred_rf = best_rf_model.predict(X_test)

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
    cv=3,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)

# Huấn luyện mô hình với tuning hyperparameters
grid_search_xgb.fit(X_train, y_train)
best_xgb_model = grid_search_xgb.best_estimator_

# Dự đoán và đánh giá cho XGBoost
y_pred_xgb = best_xgb_model.predict(X_test)

mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)


# KNN Regressor
knn = KNeighborsRegressor()
param_grid_knn = {
    'n_neighbors': [1, 50]
}
grid_search_knn = GridSearchCV(
    estimator=knn,
    param_grid=param_grid_knn,
    cv=3,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)
grid_search_knn.fit(X_train, y_train)
best_knn_model = grid_search_knn.best_estimator_

y_pred_knn = best_knn_model.predict(X_test)
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
    cv=3,
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
# Khởi tạo StandardScaler
scaler = StandardScaler()

# Chuẩn hóa dữ liệu train và test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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
    cv=3,
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


# kết quả đánh giá
results = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost', 'KNN', 'Decision Tree', 'SVR'],
    'MAE': [mae_rf, mae_xgb, mae_knn, mae_dt, mae_svr],
    'RMSE': [rmse_rf, rmse_xgb, rmse_knn, rmse_dt, rmse_svr],
    'R²': [r2_rf, r2_xgb, r2_knn, r2_dt, r2_svr]
})
print(results)


# Dữ liệu cho biểu đồ
metrics = ['MAE', 'RMSE', 'R²']
scores = {
    'MAE': [mae_rf, mae_xgb, mae_knn, mae_dt, mae_svr],
    'RMSE': [rmse_rf, rmse_xgb, rmse_knn, rmse_dt, rmse_svr],
    'R²': [r2_rf, r2_xgb, r2_knn, r2_dt, r2_svr]
}
models = ['Random Forest', 'XGBoost', 'KNN', 'Decision Tree', 'SVR']
colors = ['lightblue', 'pink', 'lightgreen', 'orange', 'violet']

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

import matplotlib.pyplot as plt

# Vẽ dự đoán 100 điểm đầu tiên của mỗi mô hình
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



