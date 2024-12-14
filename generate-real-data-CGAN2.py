import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# Download historical data for S&P 500 (2007-2009)
ticker = '^GSPC'
data = yf.download(ticker, start='2007-01-01', end='2009-12-31')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Generate financial indicators
data['MA_10'] = data['Close'].rolling(window=10).mean()
data['MA_30'] = data['Close'].rolling(window=30).mean()
data = data.dropna()

# Normalize the data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)

# Save real data with indicators
real_data = pd.DataFrame(normalized_data, columns=['Open', 'High', 'Low', 'Close', 'Volume', 'MA_10', 'MA_30'])
real_data.to_csv('real_data.csv', index=False)

print("Real data saved as 'real_data.csv'.")
