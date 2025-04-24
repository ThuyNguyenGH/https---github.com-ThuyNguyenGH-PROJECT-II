import pandas as pd
from pathlib import Path

# Đường dẫn đến file dữ liệu
data_path = Path("D:/PROJECT_II/PROJECT-II/freqtrade/user_data/data/binance/BTC_USDT-1h.feather")
df = pd.read_feather(data_path)

# Đọc 5 dòng đầu tiên
print(df.head())

import numpy as np
from sklearn.model_selection import train_test_split

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

# In ra kích thước
print("Train:", X_train.shape, "Test:", X_test.shape)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Grid Search để tìm hyperparameters tốt nhất
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)

# Huấn luyện mô hình với tuning hyperparameters
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# In ra thông số tốt nhất
print("Best Parameters:", grid_search.best_params_)

# Dự đoán và đánh giá
y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")


# Vẽ biểu đồ so sánh với 100 điểm đầu tiên
plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:100], label='Giá thực tế')
plt.plot(y_pred[:100], label='Giá dự đoán')
plt.title("Random Forest - Giá thực tế vs Dự đoán (100 giá trị đầu)")
plt.xlabel("Thời gian")
plt.ylabel("Giá BTC")
plt.legend()
plt.tight_layout()
plt.show()

#xgboostxgboost
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Khởi tạo mô hình XGBoost
xgb_model = xgb.XGBRegressor(random_state=42)

# Grid Search để tìm hyperparameters tốt nhất cho XGBoost
param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0]
}

# GridSearchCV để tìm hyperparameters tốt nhất
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

# In ra thông số tốt nhất
print("Best Parameters for XGBoost:", grid_search_xgb.best_params_)

# Dự đoán giá trị
y_pred_xgb = best_xgb_model.predict(X_test)

# Tính toán các chỉ số đánh giá
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"MAE (XGBoost): {mae_xgb:.2f}")
print(f"RMSE (XGBoost): {rmse_xgb:.2f}")
print(f"R² (XGBoost): {r2_xgb:.4f}")

# Vẽ biểu đồ so sánh với 100 điểm đầu tiên
plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:100], label='Giá thực tế')
plt.plot(y_pred_xgb[:100], label='Giá dự đoán')
plt.title("XGBoost - Giá thực tế vs Dự đoán (100 giờ đầu)")
plt.xlabel("Thời gian")
plt.ylabel("Giá BTC")
plt.legend()
plt.tight_layout()
plt.show()