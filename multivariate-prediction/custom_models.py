#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 17:49:19 2024

@author: mesabo
"""

from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Flatten, Conv1D, MaxPooling1D, GRU, Bidirectional, TimeDistributed, Attention, Input
from keras.optimizers.legacy import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from constants import (
    LSTM_MODEL, CNN_MODEL, CNN_LSTM_MODEL, CNN_GRU_MODEL, CNN_BiLSTM_MODEL,
    CNN_BiGRU_MODEL, CNN_LSTM_AE_MODEL, CNN_BiLSTM_AE_MODEL, LSTM_ATTENTION_MODEL,
    CNN_ATTENTION_MODEL, CNN_LSTM_ATTENTION_MODEL, CNN_GRU_ATTENTION_MODEL,
    CNN_BiLSTM_ATTENTION_MODEL, CNN_BiGRU_ATTENTION_MODEL, CNN_LSTM_AE_ATTENTION_MODEL,
    CNN_BiLSTM_AE_ATTENTION_MODEL
)

def train_model(model, trainX, trainY, valX, valY):
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.1, patience=8)
    ]
    print(f"[-----TRAINING MODEL {(model)}-----]\n")
    history = model.fit(trainX, trainY, epochs=20, batch_size=70, validation_data=(valX, valY), callbacks=callbacks)
    return history

def make_predictions(model, testX, scaler):
    testPredict = model.predict(testX)
    testPredict = scaler.inverse_transform(testPredict)
    return testPredict

def custom_optimizer(train_size):
    # Adam optimizer with learning rate scheduler
    optimizer =  Adam(learning_rate=1e-04, amsgrad=True)

    return optimizer



def build_model(train_size, model_type, input_shape, forecast_period):
    if model_type == "LSTM-based":
        return build_lstm_model(train_size, input_shape, forecast_period)
    elif model_type == "CNN-Based":
        return build_cnn_model(train_size, input_shape, forecast_period)
    elif model_type == "CNN-LSTM-based":
        return build_cnn_lstm_model(train_size, input_shape, forecast_period)
    elif model_type == "CNN-GRU-based":
        return build_cnn_gru_model(train_size, input_shape, forecast_period)
    elif model_type == "CNN-BiLSTM-based":
        return build_cnn_bilstm_model(train_size, input_shape, forecast_period)
    elif model_type == "CNN-BiGRU-based":
        return build_cnn_bigru_model(train_size, input_shape, forecast_period)
    elif model_type == "CNN-LSTM-AutoEncoder-based":
        return build_cnn_lstm_autoencoder_model(train_size, input_shape, forecast_period)
    elif model_type == "CNN-BiLSTM-AutoEncoder-based":
        return build_cnn_bilstm_autoencoder_model(train_size, input_shape, forecast_period)
    elif model_type == "LSTM-Attention-based":
        return build_lstm_attention_model(train_size, input_shape, forecast_period)
    elif model_type == "CNN-Attention-Based":
        return build_cnn_attention_model(train_size, input_shape, forecast_period)
    elif model_type == "CNN-LSTM-Attention-based":
        return build_cnn_lstm_attention_model(train_size, input_shape, forecast_period)
    elif model_type == "CNN-GRU-Attention-based":
        return build_cnn_gru_attention_model(train_size, input_shape, forecast_period)
    elif model_type == "CNN-BiLSTM-Attention-based":
        return build_cnn_bilstm_attention_model(train_size, input_shape, forecast_period)
    elif model_type == "CNN-BiGRU-Attention-based":
        return build_cnn_bigru_attention_model(train_size, input_shape, forecast_period)
    elif model_type == "CNN-LSTM-AutoEncoder-Attention-based":
        return build_cnn_lstm_autoencoder_attention_model(train_size, input_shape, forecast_period)
    elif model_type == "CNN-BiLSTM-AutoEncoder-Attention-based":
        return build_cnn_bilstm_autoencoder_attention_model(train_size, input_shape, forecast_period)
    else:
        raise ValueError("Invalid model type. Please choose from the available models.")

def build_lstm_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    model = Sequential()
    model.add(LSTM(100, input_shape=input_shape))
    model.add(Dense(forecast_period))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def build_cnn_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(forecast_period))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def build_cnn_lstm_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dense(forecast_period))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def build_cnn_gru_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(GRU(100))
    model.add(Dense(forecast_period))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def build_cnn_bilstm_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(100)))
    model.add(Dense(forecast_period))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def build_cnn_bigru_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(GRU(100)))
    model.add(Dense(forecast_period))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def build_cnn_lstm_autoencoder_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100, return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def build_cnn_bilstm_autoencoder_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def build_lstm_attention_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    inputs = Input(shape=input_shape)
    lstm = LSTM(100, return_sequences=True)(inputs)
    attention = Attention()(lstm)
    output = Dense(forecast_period)(attention)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def build_cnn_attention_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    inputs = Input(shape=input_shape)
    conv1d = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    max_pooling = MaxPooling1D(pool_size=2)(conv1d)
    attention = Attention()(max_pooling)
    output = Dense(forecast_period)(attention)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def build_cnn_attention_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    inputs = Input(shape=input_shape)
    conv1d = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    max_pooling = MaxPooling1D(pool_size=2)(conv1d)
    attention = Attention()(max_pooling)
    output = Dense(forecast_period)(attention)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def build_cnn_lstm_attention_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    inputs = Input(shape=input_shape)
    conv1d = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    max_pooling = MaxPooling1D(pool_size=2)(conv1d)
    lstm = LSTM(100, return_sequences=True)(max_pooling)
    attention = Attention()(lstm)
    output = Dense(forecast_period)(attention)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def build_cnn_gru_attention_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    inputs = Input(shape=input_shape)
    conv1d = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    max_pooling = MaxPooling1D(pool_size=2)(conv1d)
    gru = GRU(100, return_sequences=True)(max_pooling)
    attention = Attention()(gru)
    output = Dense(forecast_period)(attention)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def build_cnn_bilstm_attention_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    inputs = Input(shape=input_shape)
    conv1d = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    max_pooling = MaxPooling1D(pool_size=2)(conv1d)
    bilstm = Bidirectional(LSTM(100, return_sequences=True))(max_pooling)
    attention = Attention()(bilstm)
    output = Dense(forecast_period)(attention)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def build_cnn_bigru_attention_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    inputs = Input(shape=input_shape)
    conv1d = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    max_pooling = MaxPooling1D(pool_size=2)(conv1d)
    bigru = Bidirectional(GRU(100, return_sequences=True))(max_pooling)
    attention = Attention()(bigru)
    output = Dense(forecast_period)(attention)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def build_cnn_lstm_autoencoder_attention_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    inputs = Input(shape=input_shape)
    conv1d = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    max_pooling = MaxPooling1D(pool_size=2)(conv1d)
    lstm = LSTM(100, return_sequences=True)(max_pooling)
    attention = Attention()(lstm)
    output = TimeDistributed(Dense(1))(attention)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def build_cnn_bilstm_autoencoder_attention_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    inputs = Input(shape=input_shape)
    conv1d = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    max_pooling = MaxPooling1D(pool_size=2)(conv1d)
    bilstm = Bidirectional(LSTM(100, return_sequences=True))(max_pooling)
    attention = Attention()(bilstm)
    output = TimeDistributed(Dense(1))(attention)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

