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

def create_dataset(dataset, look_back, forecast_period):
    X, Y = [], []
    for i in range(len(dataset) - look_back - forecast_period + 1):
        X.append(dataset[i:(i + look_back), 0])
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
    df = pd.read_csv(url, sep=';', parse_dates={'datetime': ['Date', 'Time']}, infer_datetime_format=True, na_values=['?'])
    df.dropna(inplace=True)
    data = df.set_index('datetime')['Global_active_power'].resample('D').mean().dropna()
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data.values.reshape(-1, 1))

    trainX, trainY, valX, valY, testX, testY = split_dataset(data_normalized, look_back, forecast_period)
    
    return trainX, trainY, valX, valY, testX, testY, scaler
