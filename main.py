import pandas as pd
from pathlib import Path

# Đường dẫn đến file dữ liệu
data_path = Path("D:/PROJECT_II/PROJECT-II/freqtrade/user_data/data/binance/BTC_USDT-1h.feather")

# Đọc filefile
df = pd.read_feather(data_path)

# Đọc 5 dòng đầu tiên
print(df.head())

# Kiểm tra kiểu dữ liệu và thông tin các cột
print(df.info())

print(df.describe())

#Chuyển timestamp sang định dạng datetime
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
print(df.index)

# Tạo cột target là giá đóng cửa của giờ kế tiếp
df['target'] = df['close'].shift(-1)

# Xoá dòng cuối vì không có target
df.dropna(inplace=True)

print(df[['close', 'target']].head())

#CHIA TRAIN-TESTTEST
from sklearn.model_selection import train_test_split

# Chọn các feature (có thể mở rộng sau này)
features = ['open', 'high', 'low', 'close', 'volume']

X = df[features]
y = df['target']

# Chia tập train/test: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print(X_train.shape, X_test.shape)

