# LSTM modelmodel
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

scaler = MinMaxScaler(feature_range=(0, 1))

# Chuẩn hóa dữ liệu