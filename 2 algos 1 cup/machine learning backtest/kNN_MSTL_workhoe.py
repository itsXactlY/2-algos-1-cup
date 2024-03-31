# %% Import necessary libraries
from backtesting import Backtest, Strategy
from backtesting.test import SMA
from backtesting.lib import crossover
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Dense, Input, Dropout
from keras.models import load_model, Model
from keras.callbacks import EarlyStopping, Callback
import os
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from pprint import pprint
import pandas as pd
import numpy as np
import warnings
import time

#%% Define a function to read the data file
warnings.filterwarnings("ignore")

def _read_file(filepath):
    data = pd.read_csv(filepath, index_col=0, parse_dates=True, infer_datetime_format=True)
    
    # Convert prices to μBTC
    # data = (data / 1e6).assign(Volume=data.Volume * 1e6)
    
    # Strip datetime, adj_close values for backtest
    data.drop(columns=['Datetime'], inplace=True)
    data.drop(columns=['adj_close'], inplace=True)
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    return data

retrain_model = False
plotting = False

# Read the data file
if retrain_model:
    batchsize = 400
    data = _read_file('../../candles/BTCUSDT_1min.csv')
else:
    batchsize = 1
    data = _read_file('../../candles/binance_bars_SOL_1m.csv') # BTCUSDT_1min BTCUSDT_30min binance_bars_LINK_1m BTCUSDT_2023-3-06_1min binance_bars_RUNE_1m binance_bars_SOL_1m binance_bars_APT_1m binance_bars_EOS_1m
data = data.sort_index()

'''
Full         2017-08-17 2022-12-31 (training dataset of hope_BTC_1m_VWAP_FULL_2017-08-17_to_2022-12-31)
Bullrun      2021-02-01 2022-08-31 (initial training dataset of "hope_bullrun_1m_VPS_trained" model)
Downtrend    2021-05-09 2021-05-24
Uptrend      2021-01-27 2021-02-21
Sidetrend    2021-05-18 2021-06-10
Final        2021-04-25 2021-06-10

Unseen/Untrained Data (all after 2022-12-31)
Sideway          2023-03-06 2023-04-25
Downtrend        2023-07-21 2023-08-23


TODO :: Find and add more bull bear sideway periods after 2022-12-31
'''

# Define the Backtest date range
if retrain_model:
    start_date = "2017-08-17"
    end_date = "2022-12-31"
else:
    start_date = "2023-07-21"
    end_date = "2023-08-21"


# Resample to X-minute timeframe
# data = data.resample('5T').agg({
#     'Open': 'first',
#     'High': 'max',
#     'Low': 'min',
#     'Close': 'last',
#     'Volume': 'sum'
# }).dropna()



debug_predict = True
debug_predict_knn_lstm = True

# %% Dataframe manipulation

tp_percentage = 2 # You can change this value to any percentage you want
sl_percentage = 1.5


# Define a function for the indicators
def BBANDS(data, n_lookback, n_std):
    """Bollinger bands indicator"""
    hlc3 = (data.High + data.Low + data.Close) / 3
    mean, std = hlc3.rolling(n_lookback).mean(), hlc3.rolling(n_lookback).std()
    upper = mean + n_std*std
    lower = mean - n_std*std
    return upper, lower

def calculate_vwap(data, period):
    data['TP'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['CumVolume'] = data['Volume'].cumsum()
    data['CumPV'] = (data['TP'] * data['Volume']).cumsum()
    data['VWAP'] = data['CumPV'] / data['CumVolume']
    data.drop(['TP', 'CumVolume', 'CumPV'], axis=1, inplace=True)
    return data

def calculate_hidden_divergence(data):
    data['HiddenBullishDivergence'] = (data['Low'].shift(1) < data['Low']) & (data['VWAP'].shift(1) > data['VWAP'])
    data['HiddenBearishDivergence'] = (data['High'].shift(1) > data['High']) & (data['VWAP'].shift(1) < data['VWAP'])
    return data

# Extract necessary data and calculate features
close = data.Close.values
volume = data.Volume.values
sma10 = SMA(data.Close, 10)
sma20 = SMA(data.Close, 20)
sma50 = SMA(data.Close, 50)
sma100 = SMA(data.Close, 100)
sma200 = SMA(data.Close, 200)
upper, lower = BBANDS(data, 20, 2)

# Design matrix / independent features:

# Price-derived features
data['X_SMA10'] = (close - sma10) / close
data['X_SMA20'] = (close - sma20) / close
data['X_SMA50'] = (close - sma50) / close
data['X_SMA100'] = (close - sma100) / close
data['X_SMA200'] = (close - sma200) / close

data['X_DELTA_SMA10'] = (sma10 - sma20) / close
data['X_DELTA_SMA20'] = (sma20 - sma50) / close
data['X_DELTA_SMA50'] = (sma50 - sma100) / close
data['X_DELTA_SMA100'] = (sma100 - sma200) / close

# Indicator features
data['X_MOM'] = data.Close.pct_change(periods=2)
data['X_BB_upper'] = (upper - close) / close
data['X_BB_lower'] = (lower - close) / close
data['X_BB_width'] = (upper - lower) / close

'''
# aLca :: not needed any more as it is not needed in live usage
#      :: have the feeling the models behave different when its trained with data what is later
#      :: non existent

# Some datetime features for good measure
# data['X_day'] = data.index.dayofweek
# data['X_hour'] = data.index.hour
# Get Volume
data['Volume'] = data.Volume
'''
data['Volume'] = data.Volume


# Apply VWAP calculation
# data = calculate_vwap(data, period=20)
# # Apply hidden divergence calculation
# data = calculate_hidden_divergence(data)

data = data.dropna().astype(float)

# %% Prepare the data for modeling
# Define functions to get model design matrix and dependent variable
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

def get_clean_Xy(df):
    """Return (X, y) cleaned of NaN values"""
    X = get_X(df)
    y = get_y(df).values
    isnan = np.isnan(y)
    X = X[~isnan]
    y = y[~isnan]
    return X, y

# Prepare the data for modeling
data = data.loc[start_date:end_date]
X, y = get_clean_Xy(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, train_size=.8, random_state=0)


# %% Train the model
# Calculate the percentage of data to use for training
train_pct = 0.8  # set the percentage to use for training
n_train = int(len(data) * train_pct)
print(f"Using {train_pct * 100}% of the data ({n_train} rows) for training")


def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(32, input_shape=input_shape, return_sequences=True)) # TODO :: 32 units are too much, go back to two(?)
    model.add(Dense(1, activation='sigmoid'))

    ####################
    # WORK IN PROGRESS #
    
    # TEST LATER!      #
    # Add additional layers
    model.add(Dense(128, activation='relu'))  # Example additional dense layer
    model.add(Dropout(0.1))  # Example dropout layer
    
    # # WORK IN PROGRESS #
    # # Define the input layer
    input_layer = Input(shape=input_shape)

    # # Add existing layers from your model
    x = LSTM(128, return_sequences=True)(input_layer)

    # # Add additional layers
    x = Dense(128, activation='relu')(x)  # Example additional dense layer
    x = Dropout(0.5)(x)  # Example dropout layer

    # # Add the additional layers to the output
    output_layer = Dense(1, activation='sigmoid')(x)

    # # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    ####################

    model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy']) # (mae) loss=binary_crossentropy

    return model
# def create_lstm_model(input_shape):
#     input_layer = Input(shape=input_shape)
#     x = LSTM(128, return_sequences=True)(input_layer)
#     x = Dense(128, activation='relu')(x)
#     x = Dense(10)(x)
#     x = Dropout(0.1)(x)
#     output_layer = Dense(1, activation='sigmoid')(x)
#     model = Model(inputs=input_layer, outputs=output_layer)
#     model.compile(loss='mae', optimizer='adamax', metrics=['accuracy'])
#     return model

# def create_lstm_model(input_shape):
#     model = Sequential()
#     model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
#     model.add(Dense(1, activation='sigmoid'))
    
#     return model

def reshape_input_data(X, y, time_steps):
    num_samples, num_features = X.shape
    X_reshaped = np.zeros((num_samples - time_steps + 1, time_steps, num_features))
    y_reshaped = y[time_steps - 1:]
    for i in range(len(X_reshaped)):
        X_reshaped[i] = X[i:i + time_steps]
    return X_reshaped, y_reshaped

if debug_predict == True:
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

X_train_reshaped, y_train_reshaped = reshape_input_data(X_train, y_train, time_steps=10)  # Adjust time_steps as needed



from keras.models import Sequential
def train_lstm_model(X_train, y_train, time_steps=10, epochs=15):
    input_shape = (time_steps, X_train.shape[2])

    if retrain_model and os.path.exists(lstm_model_filename):
        model = load_lstm_model(lstm_model_filename)
    else:
        model = create_lstm_model(input_shape)
    X_train_reshaped = X_train

    print("X_train_reshaped shape:", X_train_reshaped.shape)
    print("y_train shape:", y_train.shape)

    y_train = y_train.reshape(-1, 1)

    # Define early stopping and custom progress callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    class TrainingProgressCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            # Calculate the percentage completed
            percent_complete = (epoch + 1) / epochs * 100
            print(f"{percent_complete:.2f}% complete - Loss: {logs['loss']:.4f} - Val Loss: {logs['val_loss']:.4f}")
    
    progress_callback = TrainingProgressCallback()

    # Train the model with early stopping and custom progress callback
    history = model.fit(
        X_train_reshaped, y_train, 
        epochs=epochs, 
        batch_size=batchsize,
        # validation_split=0.8,  # Use 20% of training data for validation
        # validation_steps=int(0.2 * len(X_train)),
        # validation_freq=1,
        callbacks=[early_stopping, progress_callback],  # Add the callbacks
        use_multiprocessing=True,
        # max_queue_size=10,
        workers=12,
        verbose=1,
        shuffle=False,
        validation_split=0.3
    )
    
    return model, history

def retrain_lstm_model(model_filename, new_data, epochs=150):
    # Prepare the new data for retraining
    X_new, y_new = get_clean_Xy(new_data)
    X_new_reshaped, y_new_reshaped = reshape_input_data(X_new, y_new, time_steps=10)

    # Retrain the model with the new data
    retrained_model, _ = train_lstm_model(X_new_reshaped, y_new_reshaped, epochs=epochs)

    # Save the updated model (optional)
    retrained_model.save(model_filename)
    
    return retrained_model

def save_lstm_model(model, filename):
    model.save(filename)

def load_lstm_model(filename):
    return load_model(filename)

# Load the saved LSTM model from a file if it exists, or create and train a new one

lstm_model_filename = f'debugmodel_2.h5'#'debugmodel.h5'#'hope_bullrun_1m_VWAP_new.h5'

if os.path.exists(lstm_model_filename):
    lstm_model = load_lstm_model(lstm_model_filename)
    print("Loaded LSTM model from file:", lstm_model_filename)
    if retrain_model:
        print(f"Preparing {lstm_model_filename} for Retraining over {start_date} to {end_date}")
        retrained_model = retrain_lstm_model(lstm_model_filename, data, epochs=150)
else:
    print("LSTM Model file not found. Creating a new LSTM model and training it.")
    # Unpack the trained model from the tuple
    lstm_model, _ = train_lstm_model(X_train_reshaped, y_train_reshaped, epochs=150)

    # Save the unpacked model
    lstm_model.save(lstm_model_filename)
    print('Creating a new LSTM model Done!')

# %% Predict with LSTM model
def predict_with_lstm(model, data):
    # Ensure data has 10 time steps
    if data.shape[1] < 10:
        # If data has fewer than 10 time steps, pad it with zeros
        data = np.pad(data, ((0, 0), (10 - data.shape[1], 0), (0, 0)), 'constant')
    elif data.shape[1] > 10:
        # If data has more than 10 time steps, truncate it to 10 time steps
        data = data[:, -10:, :]

    return model.predict(data)[0, 0]

N_TRAIN = n_train

class MLTrainOnceStrategy(Strategy):
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
            self.clf = KNeighborsClassifier(2)
            df = self.data.df.iloc[:N_TRAIN]
            X, y = get_clean_Xy(df)
            self.clf.fit(X, y)

    def next(self):
        # Skip the training, in-sample data
        if len(self.data) < N_TRAIN:
            return
        
        # Proceed only with out-of-sample data. Prepare some variables
        high, low, close = self.data.High, self.data.Low, self.data.Close

        # Check if there's an open position

        if self.current_position:
            print(f'TP: {self.tp_price}, SL: {self.sl_price}')
            # Check if TP or SL conditions are met
            if crossover(self.data.Close, self.tp_price):
                self.current_position.close()
                # Calculate trade return based on entry and exit times
                entry_price = self.data.df.loc[self.entry_time, 'Close']
                exit_price = self.data.Close[-1]
                trade_return = (exit_price / entry_price - 1) * 100
                print("CROSSOVER TAKEPROFIT HiT! Trade Return [%]:", trade_return)
            elif crossover(self.data.Close, self.sl_price):
                self.current_position.close()
                entry_price = self.data.df.loc[self.entry_time, 'Close']
                exit_price = self.data.Close[-1]
                trade_return = (exit_price / entry_price - 1) * 100
                print("CROSSOVER STOPLOSS HiT! Trade Return [%]:", trade_return)


        # Forecast the next movement
        X = get_X(self.data.df.iloc[-1:])
        forecast = self.clf.predict(X)[0]
        print("Forecast:", forecast)

        # Predict using the LSTM model
        lookback = 9
        
        # Get index of last row 
        end_idx = len(self.data.df) - 1
        
        # Calculate start index 
        start_idx = end_idx - lookback
        
        # Handle case where there are not enough rows
        if start_idx < 0:
            start_idx = 0
        
        # Slice 
        df_latest_data = self.data.df.iloc[start_idx:end_idx+1]
        # print(f'why i never debugged this df here after slicing?\n{df_latest_data}')
        
        ''' this snippet below is for the combined kNN model'''
        # # Ensure at least 10 rows for prediction
        # if len(df_latest_data) >= 10:
        #     X_latest_data = get_X(df_latest_data)
            
        #     # Get the number of layers dynamically
        #     num_layers = X_latest_data.shape[0]
        
        #     if num_layers >= 10:
        #         # Reshape for the LSTM model using the number of layers
        #         X_latest_data_reshaped = X_latest_data.reshape(1, num_layers, 15)
        #         lstm_forecast = predict_with_lstm(lstm_model, X_latest_data_reshaped)
        #         if debug_predict:
        #             formatted_lstm_forecast = ["{:.40f}".format(val) for val in lstm_forecast]
        #             print("LSTM Forecast:", formatted_lstm_forecast)
        #             print("X_latest_data shape:", X_latest_data.shape)
        #     else:
        #         print(f"Got only {num_layers} layers. Skipping prediction.")
        # else:
        #     print(f"Got only {len(df_latest_data)} rows. Skipping prediction.")


        # Ensure at least 10 rows for prediction
        if len(df_latest_data) >= 10:
            X_latest_data = get_X(df_latest_data)

            # Check the number of features in each time step
            num_features = X_latest_data.shape[1]
            if num_features < 10:
                # Pad the input data with zeros to match the expected shape
                num_padding = 15 - num_features
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
                    print('LSTM Forecast:', lstm_forecast)
                    print("LSTM Forecast:", formatted_lstm_forecast)
            else:
                print(f"Got only {num_samples} samples. Skipping prediction.")
        else:
            print(f"Got only {len(df_latest_data)} rows. Skipping prediction.")
    
        # Make trading decisions based on both models' predictions
        if forecast == 1 and not self.position.is_long and lstm_forecast > 0.01:
            # Store the entry time when a long position is opened
            self.entry_time = self.data.index[-1]

            self.buy(size=.1)

            print('buy')
            self.current_position = self.position  # Store the current position
            
            # Set TP and SL prices for the long position
            self.tp_price = close[-1] * (1 + tp_percentage / 100)
            self.sl_price = close[-1] * (1 - sl_percentage / 100)

        elif forecast == -1 and not self.position.is_short and lstm_forecast < 0.01:
            # Store the entry time when a short position is opened
            self.entry_time = self.data.index[-1]

            self.sell(size=.1)

            print('sell')
            self.current_position = self.position  # Store the current position
            
            # Set TP and SL prices for the short position
            self.tp_price = close[-1] * (1 - tp_percentage / 100)
            self.sl_price = close[-1] * (1 + sl_percentage / 100)


# try:
# Create a Backtest instance using the defined strategy
margin_percentage = 0.23  # 23% margin
bt = Backtest(data, MLTrainOnceStrategy, cash=100, commission=.0002, margin=margin_percentage)

# Run the backtest
stats = bt.run()

# Print the results
pprint(stats)

# Save trade statistics to a CSV file
trades = stats['_trades']
trades.to_csv(f'trades.csv', index=False)

#%% plots
if plotting:
    candlestick_data = data
    fig = go.Figure(data=[go.Candlestick(
        x=candlestick_data.index,
        open=candlestick_data['Open'],
        high=candlestick_data['High'],
        low=candlestick_data['Low'],
        close=candlestick_data['Close'],
    )])
    fig.update_layout(title=f"Candlestick Chart {start_date} - {end_date}", xaxis_title="Time", yaxis_title="Price")
    fig.show()

    # Access the entry and exit points and plot them
    # if 'trades' in stats and len(stats['_trades']):
    #     trades = stats['_trades']
    #     buy_signals = trades[trades['action'] == 'Buy']
    #     sell_signals = trades[trades['action'] == 'Sell']

    #     plt.scatter(buy_signals.index, buy_signals['price'], marker='^', color='g', label='Buy', zorder=5)
    #     plt.scatter(sell_signals.index, sell_signals['price'], marker='v', color='r', label='Sell', zorder=5)
    #     plt.legend()
