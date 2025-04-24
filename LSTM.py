# LSTM modelmodel
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

# Giả sử bạn đã có X_train, y_train, X_test, y_test từ bước trước
scaler = MinMaxScaler(feature_range=(0, 1))

# Chuẩn hóa dữ liệu
X_train_scaled = scaler.fit_transform(X_train)
y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1))
X_test_scaled = scaler.transform(X_test)
y_test_scaled = scaler.transform(y_test.reshape(-1, 1))

# Reshape dữ liệu cho LSTM: [samples, time steps, features]
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))