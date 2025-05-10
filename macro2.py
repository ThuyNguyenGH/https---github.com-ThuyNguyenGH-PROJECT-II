import pandas_datareader.data as web
import datetime

# Định nghĩa thời gian bạn muốn tải dữ liệu
start_date = datetime.datetime(2024, 9, 29)
end_date = datetime.datetime(2025, 4, 17)

# Lấy dữ liệu chỉ số CPI (Consumer Price Index)
cpi = web.DataReader('CPIAUCNS', 'fred', start_date, end_date)

# Lấy dữ liệu lãi suất Fed (Federal Funds Rate)
fed_rate = web.DataReader('FEDFUNDS', 'fred', start_date, end_date)

# Lấy dữ liệu DXY (Nominal Broad U.S. Dollar Index)
dxy = web.DataReader('DTWEXBGS', 'fred', start_date, end_date)

# In
print("CPI Data:")
print(cpi.head())
print(cpi.tail())
print("\nFed Rate Data:")
print(fed_rate.head())
print(fed_rate.tail())
print("\nDXY Data:")
print(dxy.head())
print(dxy.tail())
