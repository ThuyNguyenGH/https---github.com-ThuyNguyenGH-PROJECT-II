import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

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

# -----------------------------------------------
# Huấn luyện mô hình Random Forest
# -----------------------------------------------
# Grid Search để tìm hyperparameters tốt nhất
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
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

# -----------------------------------------------
# Huấn luyện mô hình XGBoost
# -----------------------------------------------
xgb_model = xgb.XGBRegressor(random_state=42)

# Grid Search để tìm hyperparameters tốt nhất cho XGBoost
param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
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

# -----------------------------------------------
# Tạo bảng chứa kết quả đánh giá
# -----------------------------------------------
results = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost'],
    'MAE': [mae_rf, mae_xgb],
    'RMSE': [rmse_rf, rmse_xgb],
    'R²': [r2_rf, r2_xgb]
})
print(results)

# Dữ liệu cho biểu đồ
metrics = ['MAE', 'RMSE', 'R²']
rf_scores = [mae_rf, rmse_rf, r2_rf]
xgb_scores = [mae_xgb, rmse_xgb, r2_xgb]
models = ['Random Forest', 'XGBoost']
colors = ['lightblue', 'pink']

# Vẽ 3 biểu đồ
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, metric in enumerate(metrics):
    ax = axes[i]
    values = [rf_scores[i], xgb_scores[i]]
    bars = ax.bar(models, values, color=colors)

    ax.set_title(metric)
    ax.set_ylim(0, max(values) * 1.15)

    # Hiển thị giá trị trên đầu cột
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height *0.98),
                    xytext=(0, -12),
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.tight_layout()
plt.show()