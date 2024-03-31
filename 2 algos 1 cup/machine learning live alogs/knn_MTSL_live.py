
debug_df = False
debug_predict = True
debug_predict_shape = False
batch_size = 1

# Alerts
DISCORD_ENABLED = True

#%% Candles and DATAFRAME
import ccxt
import pandas as pd
from pprint import pprint

print('CCXT Version:', ccxt.__version__)

LIMIT = 289
SYMBOL = 'LOOMUSDT'

live = ccxt.mexc3({
    'enableRateLimit': False,
    'timeout': 30000,
    'rateLimit': 2750,
    'apiKey': '',
    'secret': '',
})
live.set_sandbox_mode(False)


sandbox = ccxt.binance({
    'enableRateLimit': False,
    'timeout': 30000,
    'rateLimit': 2750,
    'apiKey': '',
    'secret': '',
    'options': {
        'defaultType': 'future',
    },
})

sandbox.set_sandbox_mode(True)
sandbox.verbose = False


def SMA(arr: pd.Series, n: int) -> pd.Series:
    """
    Returns `n`-period simple moving average of array `arr`.
    """
    return pd.Series(arr).rolling(n).mean()

def get_candles(symbol=SYMBOL, timeframe='5m', limit=LIMIT, n_lookback=20, n_std=2):
    """
    Retrieves candle data for a given symbol and timeframe.

    Args:
        symbol (str): The symbol to fetch candle data for. Defaults to the value of SYMBOL.
        timeframe (str): The timeframe for the candles. Defaults to '1m'.
        limit (int): The maximum number of candles to fetch. Defaults to the value of LIMIT.
        n_lookback (int): The number of periods to use for calculating Bollinger Bands. Defaults to 20.
        n_std (int): The number of standard deviations to use for calculating Bollinger Bands. Defaults to 2.

    Returns:
        pandas.DataFrame: A DataFrame containing the candle data with additional columns for Bollinger Bands, VWAP, hidden divergencies, SMA, and price matrix features.
    """

    if debug_df:
        print(f'Fetching candle data for {timeframe} ...')
    bars = live.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    if not bars:
        retry = True
        while retry:
            try:
                bars = live.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                retry = False
            except Exception as e:
                print(e)
    data = pd.DataFrame(bars, columns=['timestamp', 'Open', 'Low', 'High', 'Close', 'Volume'])

    '''Volume'''
    data['Volume'] = data['Volume'].astype(float)

    '''Bollinger bands indicator'''
    hlc3 = (data['High'] + data['Low'] + data['Close']) / 3
    mean, std = hlc3.rolling(n_lookback).mean(), hlc3.rolling(n_lookback).std()
    data['upper'] = mean + n_std*std
    data['lower'] = mean - n_std*std
    

    '''VWAP'''
    data['TP'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['CumVolume'] = data['Volume'].cumsum()
    data['CumPV'] = (data['TP'] * data['Volume']).cumsum()
    data['VWAP'] = data['CumPV'] / data['CumVolume']
    data.drop(['TP', 'CumVolume', 'CumPV'], axis=1, inplace=True)

    '''Hidden Divergencies on VWAP'''
    data['HiddenBullishDivergence'] = (data['Low'].shift(1) < data['Low']) & (data['VWAP'].shift(1) > data['VWAP'])
    data['HiddenBearishDivergence'] = (data['High'].shift(1) > data['High']) & (data['VWAP'].shift(1) < data['VWAP'])

    '''SMA'''
    sma10 = SMA(data['Close'], 10)
    sma20 = SMA(data['Close'], 20)
    sma50 = SMA(data['Close'], 50)
    sma100 = SMA(data['Close'], 100)
    sma200 = SMA(data['Close'], 200)

    '''Price Matrix'''
    # Price-derived features
    data['X_SMA10'] = (data['Close'] - sma10) / data['Close']
    data['X_SMA20'] = (data['Close'] - sma20) / data['Close']
    data['X_SMA50'] = (data['Close'] - sma50) / data['Close']
    data['X_SMA100'] = (data['Close'] - sma100) / data['Close']
    data['X_SMA200'] = (data['Close'] - sma200) / data['Close']

    data['X_DELTA_SMA10'] = (data['Close'] - sma20) / data['Close']
    data['X_DELTA_SMA20'] = (data['Close'] - sma50) / data['Close']
    data['X_DELTA_SMA50'] = (data['Close'] - sma100) / data['Close']
    data['X_DELTA_SMA100'] = (data['Close'] - sma200) / data['Close']

    # Indicator features
    data['X_MOM'] = data.Close.pct_change(periods=2)
    data['X_BB_upper'] = (data['upper'] - data['Close']) / data['Close']
    data['X_BB_lower'] = (data['lower'] - data['Close']) / data['Close']
    data['X_BB_width'] = (data['upper'] - data['lower']) / data['Close']

    data = data.dropna().astype(float)
    if debug_df:
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        # print(data)
        data.to_csv('debug_df.csv', index=False)

    
    return data

def update_dataframe(dataframe, limit=LIMIT, n_lookback=20, n_std=2):
    """
    Updates a given dataframe by appending new data obtained from get_candles() function.

    Args:
        dataframe (pandas.DataFrame): The original dataframe to be updated.
        limit (int, optional): The maximum number of candles to fetch. Defaults to LIMIT.
        n_lookback (int, optional): The number of previous candles to consider. Defaults to 20.
        n_std (int, optional): The number of standard deviations to use for filtering outliers. Defaults to 2.

    Returns:
        pandas.DataFrame: The updated dataframe with new data appended.
    """

    new_data = get_candles(symbol=SYMBOL, timeframe='1m', limit=limit, n_lookback=n_lookback, n_std=n_std)
    dataframe = pd.concat([dataframe, new_data], ignore_index=True)
    return dataframe

# %% Manipulate DATAFRAME for ML
from sklearn.model_selection import train_test_split
import numpy as np


def get_X(data):
    """Return model design matrix X"""
    return data.filter(like='X').values

def get_y(data):
    """Return dependent variable y"""
    y = data.Close.pct_change(48).shift(-48)  # Returns after roughly two days
    y[abs(y) < .004] = 0
    y[y > 0] = 1
    y[y < 0] = -1
    return y

def get_clean_Xy(data):
    """Return (X, y) cleaned of NaN values"""
    X = get_X(data)
    y = get_y(data).values
    isnan = np.isnan(y)
    X = X[~isnan]
    y = y[~isnan]
    return X, y

# Prepare the data for modeling

dataframe_df = get_candles(symbol=SYMBOL, timeframe='1m', limit=LIMIT, n_lookback=20, n_std=2)

X, y = get_clean_Xy(dataframe_df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

# Calculate the percentage of data to use for training
train_pct = 0.8  # set the percentage to use for training
n_train = int(len(dataframe_df) * train_pct)
print(f"Using {train_pct * 100}% of the data ({n_train} rows) for training")


#%% Load and prepare the models
from keras.models import load_model

def reshape_input_data(X, y, time_steps):
    """
    Reshape the input data for time series forecasting.
    
    Parameters:
        X (numpy.ndarray): The input data array of shape (num_samples, num_features).
        y (numpy.ndarray): The target data array of shape (num_samples,).
        time_steps (int): The number of time steps to consider for reshaping the input data.
    
    Returns:
        numpy.ndarray: The reshaped input data array of shape (num_samples - time_steps + 1, time_steps, num_features).
        numpy.ndarray: The reshaped target data array of shape (num_samples - time_steps + 1,).
    """

    num_samples, num_features = X.shape
    X_reshaped = np.zeros((num_samples - time_steps + 1, time_steps, num_features))
    y_reshaped = y[time_steps - 1:]
    for i in range(len(X_reshaped)):
        X_reshaped[i] = X[i:i + time_steps]
    return X_reshaped, y_reshaped

if debug_predict_shape == True:
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

X_train_reshaped, y_train_reshaped = reshape_input_data(X_train, y_train, time_steps=10)  # Adjust time_steps as needed


def load_lstm_model(filename):
    return load_model(filename)

import os
lstm_model_filename = f'debugmodel_2_FullBull_Run_NO_VWAP.h5'#'hope_bullrun_1m_VWAP_new.h5'
if os.path.exists(lstm_model_filename):
    lstm_model = load_lstm_model(lstm_model_filename)
    print("Loaded LSTM model from file:", lstm_model_filename)
else:
    print("LSTM Model file not found. Aborting!")
    exit()

def predict_with_lstm(model, data):
    """
    Predicts the output using a Long Short-Term Memory (LSTM) model.

    Parameters:
        model (object): The trained LSTM model.
        data (numpy.ndarray): The input data with shape (batch_size, time_steps, features).

    Returns:
        float: The predicted output value.
    """

    # Ensure data has 10 time steps
    if data.shape[1] < 10:
        # If data has fewer than 10 time steps, pad it with zeros
        data = np.pad(data, ((0, 0), (10 - data.shape[1], 0), (0, 0)), 'constant')
    elif data.shape[1] > 10:
        # If data has more than 10 time steps, truncate it to 10 time steps
        data = data[:, -10:, :]

    return model.predict(data)[0, 0]

#%% Alert imports
import requests
DISCORD_WEBHOOK_URL_GPT = 'https://discord.com/api/webhooks//'
DISCORD_WEBHOOK_URL_ZD = 'https://discord.com/api/webhooks//'
def alert(side, price=0, ltsm_forecast=0, knn_forecast=0):
    stub = f'''
    --> DEBUG ALERT <--
    Predict: {side} {SYMBOL}
    Limit Price: {price}
    Timeframe: 5m
    
    kNN  Forecast: {knn_forecast} [7 neighbors]
    MTSL Forecast: {ltsm_forecast}
    Model: {lstm_model_filename}
    
    --> DEBUG ALERT <--
=======
DISCORD_WEBHOOK_URL_ZD = 'https://discord.com/api/webhooks//--'
DISCORD_WEBHOOK_URL_CORNIX = 'https://discord.com/api/webhooks//-'
DISCORD_WEBHOOK_URL_aLca = 'https://discord.com/api/webhooks//-'

aLca_TELEGRAM = '-'
cornix_TELEGRAM = '-'
TOKEN = ''


#%% Alert Code
def alert(side, price=0, ltsm_forecast=0, knn_forecast=0):
    # Calculate stop-loss and take-profit levels
    if side == 'Long':
        stop_loss = price - (price * 0.0425)  # 4.25% below the price
        tp_1 = price + (price * 0.0125)  # 1% above the price
        tp_2 = price + (price * 0.0225)  # 2% above the price
        tp_3 = price + (price * 0.0325)  # 3% above the price
        tp_4 = price + (price * 0.0425)  # 4% above the price
        tp_5 = price + (price * 0.0525)  # 5% above the price
    else:
        stop_loss = price + (price * 0.0425)  # 4.25% below the price
        tp_1 = price - (price * 0.0125)  # 1% above the price
        tp_2 = price - (price * 0.0225)  # 2% above the price
        tp_3 = price - (price * 0.0325)  # 3% above the price
        tp_4 = price - (price * 0.0425)  # 4% above the price
        tp_5 = price - (price * 0.0525)  # 5% above the price

    
    stub = f'''
    --> DEBUG ALERT DONT USE THIS <--
    Predict: {side} {SYMBOL}
    Entry Price: {price}
    Take Profit: {tp_1} - {tp_2} - {tp_3} - {tp_4} - {tp_5}
    Stop Loss: {stop_loss}
    Leverage: 10x
    Timeframe: 1m
    
    kNN Forecast: {knn_forecast} [7 neighbors]
    LTSM Forecast: {ltsm_forecast}
    Model: {lstm_model_filename}
    
    --> DEBUG ALERT <--
    '''

    stub_tele = f'''
    {side} {SYMBOL}
    Entry: {price}
    TP: {tp_1} - {tp_2} - {tp_3} - {tp_4} - {tp_5}
    SL: {stop_loss}
    Leverage: 20x
    Forecast: {ltsm_forecast}

    '''
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    end_dc = 0 

    if DISCORD_ENABLED == True:
        chat_message = {
            "username": f"MTSL DEBUG (BINANCE - {SYMBOL} - V0.1.3-4)",
            "avatar_url": "https://cdn-icons-png.flaticon.com/512/5186/5186852.png", #"https://i.imgur.com/oF6ANhV.jpg",
            "content" : stub

        }

        start_dc = time.time()
        requests.post(DISCORD_WEBHOOK_URL_GPT, json=chat_message, headers=headers)
        requests.post(DISCORD_WEBHOOK_URL_ZD, json=chat_message, headers=headers)

        end_dc = time.time()
        print("\nTime taken sending Discord Messages: {}".format(end_dc - start_dc))


    return

#%% ML Train once -> predict -> do nothing
from sklearn.neighbors import KNeighborsClassifier

N_TRAIN = n_train

class MLTrainOnceStrategy():
    # Initialize entry time
    entry_time = None
    
    # Initialize entry price, TP price, and SL price
    entry_price = None
    tp_price = None
    sl_price = None
    current_position = None

    def init(self):
        print("Initializing strategy...")
        # Initialize the classifier if it's not already loaded
        if not hasattr(self, 'clf'):
            print("Initializing classifier...")
            # Init our model, a kNN classifier
            self.clf = KNeighborsClassifier(7)
            df = dataframe_df.iloc[:N_TRAIN]
            X, y = get_clean_Xy(df)

            self.clf.fit(X, y)

    def next(self):
        # Skip the training, in-sample data
        if len(self.df) < N_TRAIN:
            print('skipping...')
            return

        # Forecast the next movement
        X = get_X(self.df.iloc[-1:])
        forecast = self.clf.predict(X)[0]
        print(forecast)

        _forecasts = self.clf.predict(X)
        print(_forecasts)


        # Predict using the LSTM model
        lookback = 9
        
        # Get index of last row 
        end_idx = len(self.df) - 1
        
        # Calculate start index 
        start_idx = end_idx - lookback
        
        # Handle case where there are not enough rows
        if start_idx < 0:
            start_idx = 0
        
        # Slice 
        df_latest_data = self.df.iloc[start_idx:end_idx+1] 
        
        # Ensure at least 10 rows for prediction
        if len(df_latest_data) >= 10:
            X_latest_data = get_X(df_latest_data)

            # Check the number of features in each time step
            num_features = X_latest_data.shape[1]
            
            # Ensure the input data has 15 features per time step
            if num_features < 13:
                # Pad the input data with zeros to match the expected shape
                num_padding = 13 - num_features
                X_latest_data = np.pad(X_latest_data, ((0, 0), (0, num_padding)), 'constant')
            
            # Reshape the array to match the LSTM model input shape
            num_samples = X_latest_data.shape[0]
            num_time_steps = 10  # Adjust as needed based on your model
            
            if num_samples >= num_time_steps:
                X_latest_data_reshaped = X_latest_data[-num_time_steps:, :].reshape(1, num_time_steps, 13)
                lstm_forecast = predict_with_lstm(lstm_model, X_latest_data_reshaped)
                formatted_lstm_forecast = ["{:.20f}".format(val) for val in lstm_forecast]
                if debug_predict:
                    # print("kNN Prediction:", forecast)
                    print("LSTM Forecast:", formatted_lstm_forecast)
                if debug_predict_shape:
                    print("X_latest_data shape:", X_latest_data.shape)
            else:
                print(f"Got only {num_samples} samples. Skipping prediction.")
        else:
            print(f"Got only {len(df_latest_data)} rows. Skipping prediction.")
        
        try:
            if lstm_forecast > 0.01 and forecast == 1:
                # Store the entry time when a long position is opened
                # self.entry_time = datetime.now()
                alert("Long", close, formatted_lstm_forecast, forecast)
                # sandbox.create_limit_buy_order("BTCUSDT", side='buy', amount=0.01, price=close[-1])

            if lstm_forecast < 0.01 and forecast == -1:
                # Store the entry time when a short position is opened
                # self.entry_time = datetime.now()
                alert("Short", close, formatted_lstm_forecast, forecast)

                # sandbox.create_limit_sell_order("BTCUSDT", side='buy', amount=0.01, price=close[-1])
        except Exception as e:
            print("An exception occurred:", e)

#%% Scheduler
import time
def start(dataframe):
    """
    Initializes the `strategy` object, updates the `dataframe` with the latest data, 
    sets the strategy's dataframe to the updated `dataframe`, and then calls the 
    `next` method of the strategy object.

    Parameters:
        dataframe (DataFrame): The input dataframe.

    Returns:
        DataFrame: The updated dataframe.
    """
    
    strategy = MLTrainOnceStrategy()
    strategy.init()
    dataframe = update_dataframe(dataframe)  # Update the dataframe with the latest data
    strategy.df = dataframe  # Set the strategy's dataframe to the updated dataframe
    strategy.next()
    return dataframe


def wait_until_next_minute():
    # Wait until the start of the next minute
    now = time.time()
    wait_time = 60 * 5 - (now % 60)
    time.sleep(wait_time)

while True:
    try:
        # Wait until the start of the next minute
        wait_until_next_minute()
        # Update the dataframe with the latest data
        dataframe_df = update_dataframe(dataframe_df)
        # Run the bot function at the start of the full minute to the second
        dataframe_df = start(dataframe_df)
    except Exception as e:
        print("An exception occurred - {}".format(e))
