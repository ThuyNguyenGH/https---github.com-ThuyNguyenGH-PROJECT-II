import joblib
from pathlib import Path
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

loaded_rf_model = joblib.load('best_rf_model.pkl')

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
start_date = '2024-12-01'
end_date = '2025-03-15'

df_predict = df[(df.index >= start_date) & (df.index <= end_date)].copy() # Tạo bản sao

# Tạo biến mục tiêu và features (tương tự như trong quá trình huấn luyện)
df_predict['target'] = df_predict['close'].shift(-1)

df_predict['ma_3_open'] = df_predict['open'].rolling(window=3).mean()
df_predict['ma_3_high'] = df_predict['high'].rolling(window=3).mean()
df_predict['ma_3_low'] = df_predict['low'].rolling(window=3).mean()
df_predict['ma_3_close'] = df_predict['close'].rolling(window=3).mean()

df_predict['ma_6_open'] = df_predict['open'].rolling(window=6).mean()
df_predict['ma_6_high'] = df_predict['high'].rolling(window=6).mean()
df_predict['ma_6_low'] = df_predict['low'].rolling(window=6).mean()
df_predict['ma_6_close'] = df_predict['close'].rolling(window=6).mean()

df_predict['std_3_open'] = df_predict['open'].rolling(window=3).std()
df_predict['std_3_high'] = df_predict['high'].rolling(window=3).std()
df_predict['std_3_low'] = df_predict['low'].rolling(window=3).std()
df_predict['std_3_close'] = df_predict['close'].rolling(window=3).std()

# EMA
df_predict['ema_10_open'] = df_predict['open'].ewm(span=10, adjust=False).mean()
df_predict['ema_10_high'] = df_predict['high'].ewm(span=10, adjust=False).mean()
df_predict['ema_10_low'] = df_predict['low'].ewm(span=10, adjust=False).mean()
df_predict['ema_10_close'] = df_predict['close'].ewm(span=10, adjust=False).mean()

# RSI
delta_close = df_predict['close'].diff()
gain_close = (delta_close.where(delta_close > 0, 0)).rolling(14).mean()
loss_close = (-delta_close.where(delta_close < 0, 0)).rolling(14).mean()
rs_close = gain_close / loss_close
df_predict['rsi_14_close'] = 100 - (100 / (1 + rs_close))

delta_open = df_predict['open'].diff()
gain_open = (delta_open.where(delta_open > 0, 0)).rolling(14).mean()
loss_open = (-delta_open.where(delta_open < 0, 0)).rolling(14).mean()
rs_open = gain_open / loss_open
df_predict['rsi_14_open'] = 100 - (100 / (1 + rs_open))

delta_high = df_predict['high'].diff()
gain_high = (delta_high.where(delta_high > 0, 0)).rolling(14).mean()
loss_high = (-delta_high.where(delta_high < 0, 0)).rolling(14).mean()
rs_high = gain_high / loss_high
df_predict['rsi_14_high'] = 100 - (100 / (1 + rs_high))

delta_low = df_predict['low'].diff()
gain_low = (delta_low.where(delta_low > 0, 0)).rolling(14).mean()
loss_low = (-delta_low.where(delta_low < 0, 0)).rolling(14).mean()
rs_low = gain_low / loss_low
df_predict['rsi_14_low'] = 100 - (100 / (1 + rs_low))

# MACD
ema_12_open = df_predict['open'].ewm(span=12, adjust=False).mean()
ema_26_open = df_predict['open'].ewm(span=26, adjust=False).mean()
df_predict['macd_open'] = ema_12_open - ema_26_open

ema_12_high = df_predict['high'].ewm(span=12, adjust=False).mean()
ema_26_high = df_predict['high'].ewm(span=26, adjust=False).mean()
df_predict['macd_high'] = ema_12_high - ema_26_high

ema_12_low = df_predict['low'].ewm(span=12, adjust=False).mean()
ema_26_low = df_predict['low'].ewm(span=26, adjust=False).mean()
df_predict['macd_low'] = ema_12_low - ema_26_low

ema_12_close = df_predict['close'].ewm(span=12, adjust=False).mean()
ema_26_close = df_predict['close'].ewm(span=26, adjust=False).mean()
df_predict['macd_close'] = ema_12_close - ema_26_close

df_predict.dropna(inplace=True)

features = df_predict[['open', 'high', 'low', 'close', 'volume',
                   'ma_3_open', 'ma_3_high', 'ma_3_low', 'ma_3_close',
                   'ma_6_open', 'ma_6_high', 'ma_6_low', 'ma_6_close',
                   'std_3_open', 'std_3_high', 'std_3_low', 'std_3_close',
                   'ema_10_open', 'ema_10_high', 'ema_10_low', 'ema_10_close',
                   'rsi_14_open', 'rsi_14_high', 'rsi_14_low',
                   'macd_close', 'macd_open', 'macd_high', 'macd_low']]

# Thực hiện dự đoán
predictions = loaded_rf_model.predict(features)

# Tạo DataFrame kết quả
predictions_df = pd.DataFrame({'Date': features.index, 'Predicted_Price': predictions, 'Actual_Price': df_predict.loc[features.index, 'close']})

print("\nKết quả dự đoán trên dữ liệu mới:")
print(predictions_df.head())
print(predictions_df.tail())

# Vẽ biểu đồ so sánh giá thực tế và giá dự đoán
plt.figure(figsize=(15, 6))
plt.plot(predictions_df['Date'], predictions_df['Actual_Price'], label='Giá Thực Tế', color='black')
plt.plot(predictions_df['Date'], predictions_df['Predicted_Price'], label='Giá Dự Đoán (Random Forest)', color='red', linestyle='--')
plt.xlabel('Thời Gian')
plt.ylabel('Giá BTC/USDT')
plt.title('So sánh Giá Thực Tế và Giá Dự Đoán (Random Forest) trên Dữ Liệu Mới')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()