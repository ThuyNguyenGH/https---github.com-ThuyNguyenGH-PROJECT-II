import yfinance as yf

# Gold Futures, Crude Oil Futures, S&P 500, Nasdaq Composite
tickers = ['GC=F', 'CL=F', '^GSPC', '^IXIC']
data = {}

# Tải dữ liệu theo khung thời gian 1 giờ
for ticker in tickers:
    data[ticker] = yf.download(ticker, start='2024-09-25', end='2025-04-17', interval='1h')

# Lọc dữ liệu Gold, Oil, SP500, Nasdaq để phù hợp với Bitcoin
gold_data = gold_data[gold_data.index >= '2024-09-29 22:00:00+00:00']
oil_data = oil_data[oil_data.index >= '2024-09-29 22:00:00+00:00']
sp500_data = sp500_data[sp500_data.index >= '2024-09-29 22:00:00+00:00']
nasdaq_data = nasdaq_data[nasdaq_data.index >= '2024-09-29 22:00:00+00:00']

# In kết quả 5 dòng đầu tiên của giá vàng
print(data['GC=F'].head())
print(data['CL=F'].head())
print(data['^GSPC'].head())
print(data['^IXIC'].head())
