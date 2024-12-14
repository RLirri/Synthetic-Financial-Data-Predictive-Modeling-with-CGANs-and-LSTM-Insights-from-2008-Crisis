import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
import talib
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.stats import ks_2samp

# Fetching Historical Data for S&P 500
ticker = '^GSPC'
start_date = '2007-01-01'
end_date = '2009-12-31'

data = yf.download(ticker, start=start_date, end=end_date)
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
data['Adj Close'] = data['Close']  # Ensure compatibility with preprocessing scripts

# Preprocessing Data
data['price'] = data['Adj Close']
data['price_FD10'] = data['price'].shift(-10)
data['price_FD10'].fillna(data['price'].iloc[-1], inplace=True)
data['ret_D10'] = np.log(data['price'] / data['price'].shift(10))
data['ret_FD10'] = np.log(data['price_FD10'] / data['price_FD10'].shift(10))
data['label'] = np.where(data['ret_FD10'] >= 0, 1, 0)

# Adding Technical Indicators
data['mom_D28'] = talib.MOM(data['price'], timeperiod=28)
data['sma_D14'] = talib.SMA(data['price'], timeperiod=14)
data['ema_D14'] = talib.EMA(data['price'], timeperiod=14)
data['rsi_D14'] = talib.RSI(data['price'], timeperiod=14)
data['macd'], data['macdSignal'], _ = talib.MACD(data['price'])
data['bolUP_D20'], _, data['bolDOWN_D20'] = talib.BBANDS(data['price'], timeperiod=20)

# Dropping NaNs
data.dropna(inplace=True)

# Feature Selection
features = ['ret_D10', 'mom_D28', 'sma_D14', 'ema_D14', 'rsi_D14', 'macd', 'macdSignal', 'bolUP_D20', 'bolDOWN_D20']

# Train-Test Split
train_size = int(len(data) * 0.8)
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

X_train = train_data[features]
y_train = train_data['label']
X_test = test_data[features]
y_test = test_data['label']

# Scaling Features
scaler = RobustScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape for LSTM Input
def create_lstm_data(X, y, timesteps=28):
    X_lstm = []
    y_lstm = []
    for i in range(len(X) - timesteps):
        X_lstm.append(X[i:i + timesteps])
        y_lstm.append(y[i + timesteps])
    return np.array(X_lstm), np.array(y_lstm)

X_train_lstm, y_train_lstm = create_lstm_data(X_train_scaled, y_train, timesteps=28)
X_test_lstm, y_test_lstm = create_lstm_data(X_test_scaled, y_test, timesteps=28)

# Building the LSTM Model
def build_lstm_model(input_shape):
    X = Input(shape=input_shape)
    lstm = LSTM(128, return_sequences=True)(X)
    lstm = LSTM(64)(lstm)
    output = Dense(1, activation='sigmoid')(lstm)
    model = Model(inputs=X, outputs=output)
    return model

model = build_lstm_model((28, len(features)))
model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

# Training the Model
history = model.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=32, verbose=1, validation_split=0.2)

# Evaluating the Model
test_loss, test_accuracy = model.evaluate(X_test_lstm, y_test_lstm)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Predicting and Evaluating
y_pred = (model.predict(X_test_lstm) > 0.5).astype(int)
conf_matrix = confusion_matrix(y_test_lstm, y_pred)
class_report = classification_report(y_test_lstm, y_pred)

print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)

# Validating Synthetic Data with KS Test
real_sample = train_data['price']
synthetic_sample = test_data['price']  # Replace with GAN-generated data if available
ks_stat, ks_pval = ks_2samp(real_sample, synthetic_sample)
print(f"KS Test Statistic: {ks_stat}, P-Value: {ks_pval}")

# Plot Cumulative Returns
def plot_cumulative_returns(y_true, y_pred, data):
    cumulative_returns = np.cumsum((y_true.flatten() == y_pred.flatten()).astype(int))
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns, label="LSTM Model")
    plt.title("Cumulative Returns During 2008 Financial Crisis")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.show()

plot_cumulative_returns(y_test_lstm, y_pred, test_data)
