import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.layers import LSTM, Dense, Dot, Activation, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import talib

# Importing Data
# Use S&P 500 synthetic data
df = pd.read_csv("preprocessed_gan_data_with_adj_close.csv")  # Load preprocessed GAN data

# Visualizing Stock Prices
plt.figure(figsize=(10, 6))
df['Adj Close'].plot(label='S&P 500 Synthetic')
plt.legend()
plt.title('S&P 500 Synthetic Data Price Movement', fontsize=16)
plt.grid(False)
plt.show()

# Creating Features
# 10-day return direction has been chosen for prediction

sp500 = df.copy()
features = []

# Use 'Adj Close' as the price column
sp500['price'] = sp500['Adj Close']

# Add shifted forward prices
sp500['price_FD10'] = sp500['price'].shift(-10)
sp500['price_FD10'].fillna(sp500['price'].iloc[-1], inplace=True)

# Calculate returns
sp500['ret_D10'] = np.log(sp500['price'] / sp500['price'].shift(10))
sp500['ret_FD10'] = np.log(sp500['price_FD10'] / sp500['price_FD10'].shift(10))

# Create labels
sp500['label'] = np.where(sp500['ret_FD10'] >= 0, 1, 0)

# Add features
for i in [10]:
    sp500['ret_10Dlag' + str(i)] = sp500['ret_D10'].shift(i)
    features.extend(['ret_10Dlag' + str(i)])

for i in [28]:
    sp500['mom_D' + str(i)] = talib.MOM(sp500['price'], timeperiod=i)
    features.extend(['mom_D' + str(i)])

for i in [14, 50, 200]:
    sp500['sma_D' + str(i)] = talib.SMA(sp500['price'], timeperiod=i)
    sp500['ema_D' + str(i)] = talib.EMA(sp500['price'], timeperiod=i)
    sp500['rsi_D' + str(i)] = talib.RSI(sp500['price'], timeperiod=i)
    sp500['cmo_D' + str(i)] = talib.CMO(sp500['price'], timeperiod=i)
    features.extend(['sma_D' + str(i), 'ema_D' + str(i), 'rsi_D' + str(i), 'cmo_D' + str(i)])

for i in [20]:
    sp500['bolUP_D' + str(i)], middleband, sp500['bolDOWN_D' + str(i)] = talib.BBANDS(sp500['price'], timeperiod=i)
    sp500['bolBandwidth_D' + str(i)] = (sp500['bolUP_D' + str(i)] - sp500['bolDOWN_D' + str(i)]) / middleband
    features.extend(['bolUP_D' + str(i), 'bolDOWN_D' + str(i), 'bolBandwidth_D' + str(i)])

sp500['macd'], sp500['macdSignal'], _ = talib.MACD(sp500['price'])
features.extend(['macd', 'macdSignal'])

sp500.dropna(inplace=True)

def createTrainTest(stock, features, testSize=252):
    totalRecords = len(stock.index)
    test = np.arange(totalRecords - testSize, totalRecords)
    train = np.arange(0, test[0])
    X_train = stock.loc[stock.index[train], features]
    X_test = stock.loc[stock.index[test], features]
    y_train = stock.loc[stock.index[train], 'label']
    y_test = stock.loc[stock.index[test], 'label']

    return X_train, X_test, y_train.to_numpy().reshape(-1, 1), y_test.to_numpy().reshape(-1, 1)

def scaleTrainTest(X_train, X_test):
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled

def createDataLSTM(X_train, X_test, y_train, y_test, n_features, n_days=28, testSize=252):
    X_train_LSTM = np.zeros((X_train.shape[0] - n_days + 1, n_days, n_features))
    X_test_LSTM = np.zeros((testSize, n_days, n_features))
    y_train_LSTM = np.zeros((X_train.shape[0] - n_days + 1, 1))

    for i in range(X_train.shape[0] - n_days + 1):
        X_train_LSTM[i, 0:n_days] = X_train[i:i + n_days]
        y_train_LSTM[i] = y_train[i + n_days - 1]

    j = 0
    for i in range(testSize):
        if i < (n_days - 1):
            X_test_LSTM[i, 0:n_days - 1 - i] = X_train[X_train.shape[0] - n_days + 1 + i:]
            X_test_LSTM[i, n_days - 1 - i:] = X_test[0:i + 1]
        else:
            X_test_LSTM[i, 0:n_days] = X_test[j:j + n_days]
            j = j + 1
    return X_train_LSTM, X_test_LSTM, y_train_LSTM, y_test

def featureImportances(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from yellowbrick.model_selection import FeatureImportances

    rfc = RandomForestClassifier(max_depth=3, n_jobs=-1, random_state=0)
    viz1 = FeatureImportances(rfc, relative=False, labels=features)
    viz1.fit(X_train, y_train)
    viz1.show()

    abc = AdaBoostClassifier(n_estimators=100, random_state=0)
    viz2 = FeatureImportances(abc, relative=False, labels=features)
    viz2.fit(X_train, y_train)
    viz2.show()

    gbc = GradientBoostingClassifier(max_depth=3, random_state=0)
    viz3 = FeatureImportances(gbc, relative=False, labels=features)
    viz3.fit(X_train, y_train)
    viz3.show()

def plotKMeans(X_train, y_train, cols, axis_labels=None, title=None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap, BoundaryNorm

    x_min, x_max = X_train[:, cols[0]].min() - 1, X_train[:, cols[0]].max() + 1
    y_min, y_max = X_train[:, cols[1]].min() - 1, X_train[:, cols[1]].max() + 1

    class_names = ['Cluster 1', 'Cluster 2']
    color_list = ['#FFFF00', '#00AAFF']
    cmap_bold = ListedColormap(color_list)
    bnorm = BoundaryNorm(np.arange(0, 3), ncolors=2)

    plt.figure(figsize=(7, 7))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.scatter(X_train[:, cols[0]], X_train[:, cols[1]], s=100, c=y_train, cmap=cmap_bold, norm=bnorm,
                edgecolor='black', lw=1)
    plt.xlabel(axis_labels[0], fontsize=13)
    plt.ylabel(axis_labels[1], fontsize=13)

    legend_handles = []
    for i in range(0, 2):
        patch = mpatches.Patch(color=color_list[i], label=class_names[i])
        legend_handles.append(patch)
    plt.legend(handles=legend_handles, fontsize=13)
    plt.title('{}: K-Means Clustering'.format(title), fontdict={'fontsize': 16})
    plt.show()

def calcPnL(model, stock, X_test, y_test, threshold=0.5, testSize=252, stock_name=None):
    totalRecords = len(stock.index)
    test = np.arange(totalRecords - testSize, totalRecords)
    analysis = pd.DataFrame(data=stock.loc[stock.index[test], 'ret_FD10'], index=stock.index[test])
    analysis['buyAndHold'] = (stock.loc[stock.index[test], 'price'] / stock.loc[stock.index[test[0]], 'price']) - 1.0
    analysis['probUP'] = model.predict(X_test)
    analysis['betSize'] = np.where(analysis['probUP'] > threshold, 2 * analysis['probUP'] - 1, 0.0)
    analysis['dailyP&L'] = analysis['ret_FD10'] * analysis['betSize']

    profitLSTM = analysis['dailyP&L'].sum() * 100
    profitBuyHold = (analysis.loc[analysis.index[testSize - 1], 'buyAndHold']) * 100

    plt.figure(figsize=(10, 6))
    plt.plot(analysis['dailyP&L'].cumsum(), label='LSTM Attention, Return = {:.2f} %'.format(profitLSTM))
    plt.plot(analysis['buyAndHold'], label='Buy and Hold, Return = {:.2f} %'.format(profitBuyHold))
    plt.legend(fontsize=13)
    plt.title('{}: Returns Comparison (Test Set)'.format(stock_name), fontdict={'fontsize': 16})
    plt.grid(False)
    plt.show()

def plotMetrics(model, X_test, y_test, target_names, title=None):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_recall_curve, \
        roc_curve, auc

    y_scores = model.predict(X_test)
    y_pred = np.where(model.predict(X_test) >= 0.5, 1, 0)
    print("\n\033[1m\t\t\033[4m {} Confusion Matrix\033[0m\033[0m\n".format(title))
    print(confusion_matrix(y_test, y_pred))
    print("\n\033[1m\t\t\033[4m {} Classification Report\033[0m\033[0m\n".format(title))
    print(classification_report(y_test, y_pred, target_names=target_names))

    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    plt.figure(figsize=(7, 7))
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    plt.plot(precision, recall, label='Precision-Recall Curve')
    plt.xlabel('Precision', fontsize=14)
    plt.ylabel('Recall', fontsize=14)
    plt.title("{} Precision Recall Curve".format(title), fontsize=16)
    plt.grid(True)
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 7))
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.plot(fpr, tpr, lw=3, label='AUC = {:0.2f}'.format(roc_auc))
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('{} ROC curve'.format(title), fontsize=16)
    plt.legend(loc='lower right', fontsize=13)
    plt.plot([0, 1], [0, 1], lw=2, linestyle='-.')
    plt.grid(True)
    plt.show()

def plotTransitionProb(model, stock, X_test, y_test, title=None, testSize=252):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap

    y_scores = model.predict(X_test)
    y_pred = np.where(model.predict(X_test) >= 0.5, 1, 0)
    totalRecords = len(stock.index)
    test = np.arange(totalRecords - testSize, totalRecords)
    target_names = ['Wrong Prediction', 'Correct Prediction']
    c = y_test == y_pred
    color_list = ['#EEEE00', '#0000CC']
    cmap = ListedColormap(color_list)
    legend_handles = []
    for i in range(0, len(target_names)):
        patch = mpatches.Patch(color=color_list[i], label=target_names[i])
        legend_handles.append(patch)

    plt.figure(figsize=(10, 6))
    plt.scatter(stock.index[test], y_scores, c=c, cmap=cmap)
    plt.legend(loc=0, handles=legend_handles)
    plt.title('{}: Transition Probabilities for Up Moves'.format(title), fontdict={'fontsize': 16})
    plt.grid(True)

    plt.figure(figsize=(10, 6))
    plt.scatter(stock.index[test], 1 - y_scores, c=c, cmap=cmap)
    plt.legend(loc=0, handles=legend_handles)
    plt.title('{}: Transition Probabilities for Down Moves'.format(title), fontdict={'fontsize': 16})
    plt.grid(True)

def model(inputDays, unitsPreLSTM, unitsPostLSTM, unitsDenseAttention, featuresLen, name=None):
    X = Input(shape=(inputDays, featuresLen), name='Input')
    a = LSTM(units=unitsPreLSTM, return_sequences=True, dropout=0.05, name='PreAttention_LSTM')(X)

    e1 = Dense(unitsDenseAttention, activation="relu", name='Dense1_Attention')(a)
    e2 = Dense(1, activation="relu", name='Dense2_Attention')(e1)
    alphas = Activation(lambda x: K.softmax(x, axis=1), name='attention_weights')(e2)
    context = Dot(axes=1, name='context')([alphas, a])

    s = LSTM(unitsPostLSTM, dropout=0.05, name='PostAttention_LSTM')(context)
    output = Dense(1, activation='sigmoid', name='Output')(s)

    model = Model(inputs=X, outputs=output, name=name)
    return model

X_train, X_test, y_train, y_test = createTrainTest(sp500, features)
X_train_scaled, X_test_scaled = scaleTrainTest(X_train, X_test)
X_train_LSTM, X_test_LSTM, y_train_LSTM, y_test_LSTM = createDataLSTM(
    X_train_scaled, X_test_scaled, y_train, y_test, len(features), n_days=28, testSize=252)

# Feature Importance Visualization
featureImportances(X_train_scaled, y_train)

# K-Means Clustering Visualization
cols = [0, 1]  # Adjust these indices based on desired features
axis_labels = ['Feature 1', 'Feature 2']  # Adjust feature names
plotKMeans(X_train_scaled, y_train, cols, axis_labels, title='K-Means Clustering')

inputDays = 28
unitsPreLSTM = 128
unitsPostLSTM = 32
unitsDenseAttention = 32

modelSP500 = model(inputDays, unitsPreLSTM, unitsPostLSTM, unitsDenseAttention, len(features), name='model_SP500')
modelSP500.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
modelSP500.fit(X_train_LSTM, y_train_LSTM, epochs=50, batch_size=28, verbose=2)

modelSP500.evaluate(X_test_LSTM, y_test_LSTM)

# Metrics and Evaluation Plots
target_names = {0: "Down Move", 1: "Up Move"}
plotMetrics(modelSP500, X_test_LSTM, y_test_LSTM, list(target_names.values()), title="S&P 500 Metrics")

# Transition Probabilities Visualization
plotTransitionProb(modelSP500, sp500, X_test_LSTM, y_test_LSTM, title="S&P 500 Transition Probabilities", testSize=252)

calcPnL(modelSP500, sp500, X_test_LSTM, y_test_LSTM, threshold=0.52, stock_name='S&P 500 Synthetic')
