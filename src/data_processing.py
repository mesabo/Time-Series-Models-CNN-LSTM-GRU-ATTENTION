#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 17:50:00 2024

@author: mesabo
"""

# data_processing.py

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
import json
from constants import (DATASET_FEATURES_PATH, ELECTRICITY_DATASET_PATH,
                       ELECTRICITY, WATER, WIND, GOLD)


def fill_missing_data(data, meth=2):
    if meth == 1:
        # 2. Imputation with Simple Statistics
        # Replace missing values with the mean for numeric columns
        data.fillna(data.mean(), inplace=True)
    elif meth == 2:
        # 3. Forward or Backward Fill (Time Series Data)
        data.sort_values(by="datetime", inplace=True)
        data.fillna(method="ffill", inplace=True)  # Forward fill
    elif meth == 3:
        # 4. Interpolation
        # Linear interpolation for numeric columns
        data.interpolate(method="linear", inplace=True)
    else:
        # 1. Dropping Rows or Columns
        # Drop rows with any missing values
        data = data.dropna()

    return data


def read_features(features_path, key_name):
    with open(features_path, "r") as file:
        data_features = json.load(file)

    if key_name in data_features:
        features = data_features[key_name]
        return features
    else:
        raise KeyError(f"Key '{key_name}' not found in the JSON file.")


def create_dataset(dataset, look_back, forecast_period):
    X, Y = [], []
    for i in range(len(dataset) - look_back - forecast_period + 1):
        X.append(dataset[i:(i + look_back), :])  # Adjust for multiple features
        Y.append(
            dataset[(i + look_back):(i + look_back + forecast_period), 0])  # Assuming first column is output feature
    return np.array(X), np.array(Y)

def time_warping(series, sigma=0.2):
    n = len(series)
    time_stretching = np.cumsum(np.random.randn(n) * sigma)
    time_stretching -= time_stretching.min()
    time_stretching /= time_stretching.max()
    new_time = np.arange(n)
    warped_time = new_time + time_stretching * (n - 1)
    warped_series = interp1d(warped_time, series, bounds_error=False, fill_value="extrapolate")(new_time)
    return warped_series

def split_dataset(data_normalized, look_back, forecast_period):
    train_size = int(len(data_normalized) * 0.7)
    test_size = len(data_normalized) - train_size

    train = data_normalized[:train_size]
    test = data_normalized[-test_size:]

    trainX, trainY = create_dataset(train, look_back, forecast_period)
    testX, testY = create_dataset(test, look_back, forecast_period)

    return trainX, trainY, testX, testY


def load_dataset(dataset_type='ELECTRICITY', period='D'):
    if dataset_type == ELECTRICITY:
        dataset = pd.read_csv(ELECTRICITY_DATASET_PATH, sep=';', parse_dates={'datetime': ['Date', 'Time']},
                              na_values=['?'])
        df = fill_missing_data(dataset, meth=2)
        selected_features = read_features(DATASET_FEATURES_PATH, dataset_type)

        data = df.set_index('datetime')[selected_features].resample(period).mean().dropna()
        # Separate features and target variable
        features = data.drop(columns=selected_features[0]).values
        target = data[selected_features[0]].values.reshape(-1, 1)

    return features, target


def preprocess_and_split_dataset(url, period, look_back, forecast_period):
    # Separate features and target variable
    features, target = load_dataset(url, period)

    # Normalize features
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler_features.fit_transform(features)

    # Normalize target variable
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaled_target = scaler_target.fit_transform(target)

    # Combine scaled features and target variable
    scaled_dataset = np.concatenate((scaled_features, scaled_target), axis=1)

    # Apply time warping to the features (excluding the target variable)
    warped_features = np.apply_along_axis(time_warping, axis=0, arr=scaled_dataset[:, :-1])

    # Combine warped features with target variable
    warped_dataset = np.concatenate((warped_features, scaled_dataset[:, -1].reshape(-1, 1)), axis=1)

    trainX, trainY, testX, testY = split_dataset(warped_dataset, look_back, forecast_period)

    return trainX, trainY, testX, testY, scaler_target
