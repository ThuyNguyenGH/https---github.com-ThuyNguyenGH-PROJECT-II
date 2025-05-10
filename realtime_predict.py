import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Đường dẫn đến dữ liệu trong Freqtrade
data_path = Path("D:/PROJECT_II/freqtrade/user_data/data/binance/BTC_USDT-1h.feather")

# Tải mô hình đã huấn luyện
best_rf_model = joblib.load('random_forest_model.pkl')

# Hàm lấy dữ liệu từ Freqtrade
def get_freqtrade_data():
    # Đọc dữ liệu từ file feather của Freqtrade
    df = pd.read_feather(data_path)
    
    # Chuyển cột 'date' thành kiểu datetime và thiết lập làm index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Tạo thêm các đặc trưng kỹ thuật
    df['ma_3'] = df['close'].rolling(window=3).mean()    # Trung bình 3 giờ
    df['ma_6'] = df['close'].rolling(window=6).mean()    # Trung bình 6 giờ
    df['std_3'] = df['close'].rolling(window=3).std()    # Độ lệch chuẩn 3 giờ
    
    # Loại bỏ các giá trị NaN
    df.dropna(inplace=True)
    
    return df

# Hàm dự đoán giá theo thời gian thực
def predict_real_time():
    while True:
        # Lấy dữ liệu từ Freqtrade
        df = get_freqtrade_data()
        
        # Chọn các đặc trưng cần thiết cho mô hình
        features_new = df[['close', 'volume', 'ma_3', 'ma_6', 'std_3']]
        
        # Dự đoán giá từ mô hình Random Forest
        y_pred_new = best_rf_model.predict(features_new)
        
        # Dự đoán giá BTC cho giờ tiếp theo
        predicted_price = y_pred_new[-1]
        
        # Lấy thời gian hiện tại
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # In ra dự đoán giá cùng thời gian hiện tại
        print(f"Dự đoán giá BTC vào {current_time}: {predicted_price}")
        
        # Vẽ biểu đồ so sánh dự đoán với giá thực tế
        plt.figure(figsize=(15, 6))
        plt.plot(df['close'], label='Giá thực tế', color='black', linewidth=2)
        plt.plot(df.index[-1], predicted_price, marker='o', label=f'Dự đoán {predicted_price}', color='red')
        plt.title('Dự đoán giá BTC theo thời gian thực', fontsize=14)
        plt.xlabel('Thời gian', fontsize=12)
        plt.ylabel('Giá BTC-USDT', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Chờ 1 giờ trước khi lấy dữ liệu và dự đoán lại
        time.sleep(3600)

# Chạy dự đoán giá theo thời gian thực
predict_real_time()

