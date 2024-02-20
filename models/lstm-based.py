#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 17:36:15 2024

@author: mesabo
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Function to split dataset into train, validation, and test sets
def split_dataset(data_normalized, look_back, forecast_period):
    train_size = int(len(data_normalized) * 0.7)
    val_size = int(len(data_normalized) * 0.15)
    test_size = len(data_normalized) - train_size - val_size
    
    train = data_normalized[:train_size]
    val = data_normalized[train_size:train_size+val_size]
    test = data_normalized[-test_size:]
    
    trainX, trainY = create_dataset(train, look_back, forecast_period)
    valX, valY = create_dataset(val, look_back, forecast_period)
    testX, testY = create_dataset(test, look_back, forecast_period)
    
    return trainX, trainY, valX, valY, testX, testY


# Function to preprocess dataset
def preprocess_dataset(url):
    # Load dataset
    df = pd.read_csv(url, sep=';', parse_dates={'datetime': ['Date', 'Time']}, infer_datetime_format=True, na_values=['?'])
    # Handle missing values
    df.dropna(inplace=True)
    # Convert to time series format
    data = df.set_index('datetime')['Global_active_power'].resample('D').mean().dropna()
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data.values.reshape(-1, 1))
    return data_normalized, scaler

# Function to create dataset with look back and forecast period
def create_dataset(dataset, look_back=1, forecast_period=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - forecast_period + 1):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[(i + look_back):(i + look_back + forecast_period), 0])
    return np.array(X), np.array(Y)

# Function to build LSTM model
def build_lstm_model(input_shape, forecast_period):
    model = Sequential()
    model.add(LSTM(100, input_shape=input_shape))
    model.add(Dense(forecast_period))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Function to train the model
def train_model(model, trainX, trainY, valX, valY):
    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.1, patience=2)
    ]
    history = model.fit(trainX, trainY, epochs=20, batch_size=70, validation_data=(valX, valY), callbacks=callbacks)
    return history

# Function to make predictions
def make_predictions(model, testX, scaler):
    # Make predictions
    testPredict = model.predict(testX)
    # Invert predictions to original scale
    testPredict = scaler.inverse_transform(testPredict)
    return testPredict

# Function to evaluate the model
def evaluate_model(testY, testPredict):
    # Calculate MAE and RMSE
    mae = mean_absolute_error(testY, testPredict)
    rmse = mean_squared_error(testY, testPredict, squared=False)
    return mae, rmse

# Function to plot losses
def plot_losses(history):
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Function to save the model
def save_model(model, model_name):
    model.save("../input/"+model_name + '.h5')

# Main function to run the process
def main():
    url = "../input/household_power_consumption.txt"
    look_back = 30  # Look back 30 days
    forecast_period = 7  # Forecast 7 days ahead
    data_normalized, scaler = preprocess_dataset(url)
    trainX, trainY, valX, valY, testX, testY = split_dataset(data_normalized, look_back, forecast_period)
    
    # LSTM-based model
    lstm_model = build_lstm_model((look_back, 1), forecast_period)
    lstm_history = train_model(lstm_model, trainX, trainY, valX, valY)
    lstm_testPredict = make_predictions(lstm_model, testX, scaler)
    lstm_mae, lstm_rmse = evaluate_model(testY, lstm_testPredict)
    plot_losses(lstm_history)
    save_model(lstm_model, "lstm_model")

if __name__ == "__main__":
    main()
