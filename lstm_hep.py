#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 00:01:43 2024

@author: mesabo
"""
# system 
import os 
import sys

# data processing
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

# model processing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error,r2_score

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler
)

# model building
from keras.models import Sequential
from keras.layers import (LSTM, Conv1D, MaxPooling1D, Flatten, Dense, RepeatVector, TimeDistributed)
from keras.optimizers.legacy import Adam, SGD


"""
-------------------------------------------------------------------------------
-----------------------------------S T A R T ----------------------------------
-------------------------------------------------------------------------------
"""


""" Functions """

# Checking Nans and duplicates in each columns
def check_NaN_values(data):
    print(f'Number of Nans in each column :\n{data.isnull().sum()}\n')
    print(f'\nNumber of duplicates in the dataframe : {data.duplicated().sum()}')
    return


def feat_corr(data):
    corr = data.corr()
    plt.figure(figsize=(15,12))
    #plot heat map
    g=sns.heatmap(corr,annot=True,cmap="RdYlGn", vmin=-1, vmax=1)
    plt.title('Feature Correlation')
    
    return plt.show()

def visual_cols(data):
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15,15))
    
    # select the columns to plot
    columns_to_plot = ['Global_active_power', 'Global_reactive_power', 'Global_intensity', 'Voltage', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    
    # loop through the subplots and plot each column
    for i, ax in enumerate(axes.flat):
        if i < len(columns_to_plot):
            ax.plot(data.index, data[columns_to_plot[i]])
            ax.set_title(columns_to_plot[i])
        else:
            ax.set_visible(False)
    
    plt.tight_layout()  # adjust the spacing between subplots
    plt.show()  # display the plot
    return

def fill_nan_value(data):
    for j in range(0,7):
        data.iloc[:,j]=data.iloc[:,j].fillna(data.iloc[:,j].mean())
    print(f'Null data:\n{data.isnull().sum()}')
    return data

""" Loading Dataset """

file_path = './input/household_power_consumption_days.csv'

dataset = pd.read_csv(file_path, 
                infer_datetime_format=True,
                low_memory=False, na_values=['nan','?'],
                index_col='datetime')


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    dff = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(dff.shift(-i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(dff.shift(-i))
        if i==0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1)) for j in range(n_vars)]

        agg = pd.concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg

def scale_data(data, scaler):
    scaled = scaler.fit_transform(data)
    return scaled


def revert_scaled_data(data, scaler):
    pass


def model_lstm(in_shape):
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=in_shape))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def visual_model(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    return

""" Visualizing Dataset Infos and Stats"""

dataset.describe().T
check_NaN_values(dataset)
feat_corr(dataset)
dataset.plot()

dataset.head()
dataset.drop(['sub_metering_4'], axis= 1, inplace = True)

dataset.info()

visual_cols(dataset)


""" Preprocessing Dataset"""

pro_data = fill_nan_value(dataset)

pro_data.head()


""" Processing Data """

scaler = MinMaxScaler(feature_range=(0, 1))

scaled_data = scale_data(pro_data.values, scaler)


# frame as supervised learning
"""reframed_data = series_to_supervised(scaled_data, 1, 1)
reframed_data.head()

# drop columns we don't want to predict
reframed_data = reframed_data.iloc[:, :8]
reframed_data.head()
"""

# X contains the features (input variables), and y contains the target variable (output variable)
reframed_data_values = scaled_data
X = reframed_data_values[:, :-1]  # Features (all columns except the last one)
y = reframed_data_values[:, -1]   # Target variable (last column)

# Splitting data and remove the last row of Train for consistent size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
#X_train = np.delete(X_train, -1, axis=0)
#y_train = np.delete(y_train, -1)

train_X = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
test_X = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

#train_x = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
#test_x = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))



""" Applying Models """

input_shape = (train_X.shape[1], train_X.shape[2])

lstm_model = model_lstm(in_shape=input_shape)

fitted_lstm_model = lstm_model.fit(train_X, y_train, epochs=2, batch_size=64, validation_data=(test_X, y_test), verbose=1, shuffle=False)

visual_model(fitted_lstm_model)


predicted_model = lstm_model.predict(test_X)


# Inverting scaled data back to original scale
reverted_predicted_values = scaler.inverse_transform(predicted_model)

# Calculating MSE, RMSE, and MAE
mse_score = mean_squared_error(y_test, reverted_predicted_values)
rmse_score = np.sqrt(mse_score)
mae_score = np.mean(np.abs(y_test - reverted_predicted_values))

print("Mean Squared Error (MSE):", mse_score)
print("Root Mean Squared Error (RMSE):", rmse_score)
print("Mean Absolute Error (MAE):", mae_score)








