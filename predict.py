import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

# Đường dẫn đến file dữ liệu
data_path = Path("D:/PROJECT_II/freqtrade/user_data/data/binance/BTC_USDT-1h.feather")
df = pd.read_feather(data_path)

print("Dữ liệu:")
print(df.head())  # In 5 dòng đầu của dữ liệu

# Chuyển cột 'date' thành kiểu datetime
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Lọc dữ liệu trong khoảng thời gian từ 1/9/2023 đến 1/5/2024
start_date = '2024-09-01'
end_date = '2025-05-01'

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

# Tải mô hình đã huấn luyện
best_rf_model = joblib.load('random_forest_model.pkl')

# Tạo features_new cho dữ liệu mới (bạn đã lọc trước đó)
features_new = df[['close', 'volume', 'ma_3', 'ma_6', 'std_3', 'ema_10', 'rsi_14', 'macd']].tail(100)

# Dự đoán giá từ mô hình
y_pred_new = best_rf_model.predict(features_new)

# Tạo DataFrame chứa kết quả dự đoán với thời gian
predicted_df = pd.DataFrame({
    'Datetime': df.index[-100:],  # 100 giờ cuối
    'Predicted_Price': y_pred_new
})

print("Dự đoán giá BTC kèm ngày giờ (100 giờ cuối):")
print(predicted_df.to_string(index=False))

# Vẽ biểu đồ so sánh dự đoán với giá thực tế
plt.figure(figsize=(15, 6))
plt.plot(df['close'].tail(100).values, label='Giá thực tế', color='black', linewidth=2)  # Giá thực tế 100 giờ cuối
plt.plot(y_pred_new, label='Dự đoán từ Random Forest', linestyle='--')  # Dự đoán giá
plt.title('So sánh dự đoán với Giá thực tế (100 giờ cuối)', fontsize=14)
plt.xlabel('Thời gian (index)', fontsize=12)
plt.ylabel('Giá BTC-USDT', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
