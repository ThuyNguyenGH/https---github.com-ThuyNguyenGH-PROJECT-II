import pandas as pd
from pathlib import Path

data_path = Path("D:/PROJECT_II/PROJECT-II/freqtrade/user_data/data/binance/BTC_USDT-1h.feather")

# Đọc dữ liệu .feather
df = pd.read_feather(data_path)

# Hiển thị 5 dòng đầu tiên
print(df.head())

# Kiểm tra thông tin về các cột và kiểu dữ liệu
print(df.info())

# Kiểm tra các giá trị thống kê như mean, min, max, std...
print(df.describe())

# Kiểm tra các giá trị thiếu trong dữ liệu
print(df.isnull().sum())

# Tạo nhãn (label) là giá đóng cửa tiếp theo
df['future_close'] = df['close'].shift(-1)

# Xử lý datetime thành đặc trưng
df['hour'] = df['date'].dt.hour
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday


df = df.dropna()

# Chọn đặc trưng (feature)
features = ['open', 'high', 'low', 'close', 'volume', 'hour', 'day', 'weekday']
X = df[features]
y = df['future_close']
print(X.head())
print(y.head())
