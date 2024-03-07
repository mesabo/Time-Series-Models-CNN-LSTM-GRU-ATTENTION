#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 17:50:00 2024

@author: mesabo
"""

# data_processing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def fill_missing_data(data, meth=2):
    if meth == 1:
        # 2. Imputation with Simple Statistics
        # Replace missing values with the mean for numeric columns
        data.fillna(data.mean(), inplace=True)
    elif meth == 2:
        # 3. Forward or Backward Fill (Time Series Data)
        data.sort_values(by='datetime', inplace=True)
        data.fillna(method='ffill', inplace=True)  # Forward fill
    elif meth == 3:
        # 4. Interpolation
        # Linear interpolation for numeric columns
        data.interpolate(method='linear', inplace=True)
    else:
        # 1. Dropping Rows or Columns
        # Drop rows with any missing values
        data = data.dropna()

    return data


def create_dataset(dataset, look_back, forecast_period):
    X, Y = [], []
    for i in range(len(dataset) - look_back - forecast_period + 1):
        X.append(dataset[i:(i + look_back), :])
        Y.append(dataset[(i + look_back):(i + look_back + forecast_period), 0])
    return np.array(X), np.array(Y)


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


def preprocess_and_split_dataset(url, look_back, forecast_period):
    df = pd.read_csv(url, sep=';', parse_dates={'datetime': [
                     'Date', 'Time']}, na_values=['?'])
    df = fill_missing_data(df, meth=2)
    #selected_features = ['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    selected_features = ['Global_active_power']

    data = df.set_index('datetime')[
        selected_features].resample('D').mean().dropna()
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data.values.reshape(-1, 1))

    trainX, trainY, valX, valY, testX, testY = split_dataset(
        data_normalized, look_back, forecast_period)

    return trainX, trainY, valX, valY, testX, testY, scaler
