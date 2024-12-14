import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import RobustScaler

# Load GAN synthetic data
gan_data = pd.read_csv("/Users/ruixinhuang/Desktop/CGAN-LSTM/synthetic_data_conditional.csv")

# Debug: Check for invalid values in the 'Close' column
print("Initial data issues:")
print(gan_data['Close'].describe())
print(f"NaN values in 'Close': {gan_data['Close'].isna().sum()}")

# Fix invalid 'Close' values
gan_data['price'] = gan_data['Close'].clip(lower=1e-6)  # Ensure no zero or negative values
gan_data['price_FD10'] = gan_data['price'].shift(-10)

# Debug: Verify price columns
print("\nPrice column issues after fixing:")
print(gan_data[['price', 'price_FD10']].describe())

# Calculate returns
gan_data['ret_D10'] = np.where(
    gan_data['price'].shift(10) > 0,
    np.log(gan_data['price'] / gan_data['price'].shift(10)),
    np.nan
)
gan_data['ret_FD10'] = np.where(
    gan_data['price_FD10'] > 0,
    np.log(gan_data['price_FD10'] / gan_data['price_FD10'].shift(10)),
    np.nan
)
gan_data['label'] = np.where(gan_data['ret_FD10'] >= 0, 1, 0)

# Debug: Check for NaN in calculated returns
print("\nReturn calculation issues:")
print(f"NaN values in 'ret_D10': {gan_data['ret_D10'].isna().sum()}")
print(f"NaN values in 'ret_FD10': {gan_data['ret_FD10'].isna().sum()}")

# Add technical indicators
gan_data['mom_D28'] = talib.MOM(gan_data['price'], timeperiod=28)
gan_data['sma_D14'] = talib.SMA(gan_data['price'], timeperiod=14)
gan_data['ema_D14'] = talib.EMA(gan_data['price'], timeperiod=14)
gan_data['rsi_D14'] = talib.RSI(gan_data['price'], timeperiod=14)
gan_data['macd'], gan_data['macdSignal'], _ = talib.MACD(gan_data['price'])
gan_data['bolUP_D20'], _, gan_data['bolDOWN_D20'] = talib.BBANDS(gan_data['price'], timeperiod=20)

# Add 'Adj Close' column for LSTM script compatibility
gan_data['Adj Close'] = gan_data['price']  # Use 'price' as equivalent to 'Adj Close'

# Drop rows with NaN values
gan_data.dropna(inplace=True)

# Normalize features
features = ['ret_D10', 'mom_D28', 'sma_D14', 'ema_D14', 'rsi_D14', 'macd', 'macdSignal', 'bolUP_D20', 'bolDOWN_D20']
scaler = RobustScaler()
gan_data[features] = scaler.fit_transform(gan_data[features])

# Save preprocessed GAN data
output_path = "/Users/ruixinhuang/Desktop/CGAN-LSTM/preprocessed_gan_data_with_adj_close.csv"
gan_data.to_csv(output_path, index=False)
print(f"Preprocessed GAN data with 'Adj Close' saved as '{output_path}'.")
