#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 17:49:19 2024

@author: mesabo
"""

from keras.models import Sequential, Model
from keras.layers import (LSTM, Dense, Flatten, Conv1D, MaxPooling1D, GRU, 
                          Bidirectional, TimeDistributed, Attention, Input,
                          Reshape, RepeatVector,Masking, Concatenate, dot,
                          Permute, Dropout, BatchNormalization)
from keras.optimizers.legacy import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from constants import (
    LSTM_MODEL, GRU_MODEL , CNN_MODEL, BiLSTM_MODEL , BiGRU_MODEL,
    LSTM_ATTENTION_MODEL, GRU_ATTENTION_MODEL , CNN_ATTENTION_MODEL , 
    BiLSTM_ATTENTION_MODEL , BiGRU_ATTENTION_MODEL , 
    CNN_LSTM_MODEL, CNN_GRU_MODEL, CNN_BiLSTM_MODEL, CNN_BiGRU_MODEL,
    CNN_LSTM_ATTENTION_MODEL , CNN_GRU_ATTENTION_MODEL , 
    CNN_BiLSTM_ATTENTION_MODEL , CNN_BiGRU_ATTENTION_MODEL , 
    #CNN_LSTM_ATTENTION_LSTM_MODEL , CNN_GRU_ATTENTION_GRU_MODEL ,
    #CNN_BiLSTM_ATTENTION_BiLSTM_MODEL , CNN_BiGRU_ATTENTION_BiGRU_MODEL , 
    CNN_ATTENTION_LSTM_ATTENTION_MODEL,CNN_ATTENTION_GRU_ATTENTION_MODEL, 
    CNN_ATTENTION_BiLSTM_ATTENTION_MODEL, CNN_ATTENTION_BiGRU_ATTENTION_MODEL,
    CNN_ATTENTION_LSTM_MODEL , CNN_ATTENTION_GRU_MODEL , 
    CNN_ATTENTION_BiLSTM_MODEL , CNN_ATTENTION_BiGRU_MODEL,
)

def train_model(model, trainX, trainY, valX, valY):
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.1, patience=8)
    ]
    print(f"[---------------TRAINING MODEL---------------]\n")
    history = model.fit(trainX, trainY, epochs=20, batch_size=70, validation_data=(valX, valY), callbacks=callbacks)
    return history

def make_predictions(model, testX, testY, scaler):
    testPredict = model.predict(testX)
    testPredict = scaler.inverse_transform(testPredict)
    testFeature = scaler.inverse_transform(testX)
    testOutput = scaler.inverse_transform(testY)
    return testPredict, testFeature, testOutput

def custom_optimizer(train_size):
    # Adam optimizer with learning rate scheduler
    optimizer =  Adam(learning_rate=1e-04, amsgrad=True)

    return optimizer



def build_model(train_size, model_type, input_shape, forecast_period):
    #-----------------------------Simple models-------------------------------
    if model_type == LSTM_MODEL:
        return build_lstm_model(train_size, input_shape, forecast_period)
    elif model_type == GRU_MODEL:
        return build_gru_model(train_size, input_shape, forecast_period)
    elif model_type == CNN_MODEL:
        return build_cnn_model(train_size, input_shape, forecast_period)
    #-----------------------------Simple Bi models-------------------------------
    elif model_type == BiLSTM_MODEL:
        return build_bilstm_model(train_size, input_shape, forecast_period)
    elif model_type == BiGRU_MODEL:
        return build_bigru_model(train_size, input_shape, forecast_period)
    #-----------------------------Simple + Attention models-------------------------------
    elif model_type == LSTM_ATTENTION_MODEL:
        return build_lstm_attention_model(train_size, input_shape, forecast_period)
    elif model_type == GRU_ATTENTION_MODEL:
        return build_gru_attention_model(train_size, input_shape, forecast_period)
    elif model_type == CNN_ATTENTION_MODEL:
        return build_cnn_attention_model(train_size, input_shape, forecast_period)
    #-----------------------------Bi + Attention models-------------------------------
    elif model_type == BiLSTM_ATTENTION_MODEL:
        return build_bilstm_attention_model(train_size, input_shape, forecast_period)
    elif model_type == BiGRU_ATTENTION_MODEL:
        return build_bigru_attention_model(train_size, input_shape, forecast_period)
    #-----------------------------Hybrid models-------------------------------
    elif model_type == CNN_LSTM_MODEL:
        return build_cnn_lstm_model(train_size, input_shape, forecast_period)
    elif model_type == CNN_GRU_MODEL:
        return build_cnn_gru_model(train_size, input_shape, forecast_period)
    elif model_type == CNN_BiLSTM_MODEL:
        return build_cnn_bilstm_model(train_size, input_shape, forecast_period)
    elif model_type == CNN_BiGRU_MODEL:
        return build_cnn_bigru_model(train_size, input_shape, forecast_period)
    #-----------------------------Hybrid + Attention models-------------------------------    
    elif model_type == CNN_LSTM_ATTENTION_MODEL:
        return build_cnn_lstm_attention_model(train_size, input_shape, forecast_period)
    elif model_type == CNN_GRU_ATTENTION_MODEL:
        return build_cnn_gru_attention_model(train_size, input_shape, forecast_period)
    elif model_type == CNN_BiLSTM_ATTENTION_MODEL:
        return build_cnn_bilstm_attention_model(train_size, input_shape, forecast_period)
    elif model_type == CNN_BiGRU_ATTENTION_MODEL:
        return build_cnn_bigru_attention_model(train_size, input_shape, forecast_period)
    #-----------------------------Deep Hybrid + Attention models-------------------------------    
    elif model_type == CNN_ATTENTION_LSTM_MODEL:
        return build_cnn_attention_lstm_model(train_size, input_shape, forecast_period)
    elif model_type == CNN_ATTENTION_GRU_MODEL:
        return build_cnn_attention_gru_model(train_size, input_shape, forecast_period)
    elif model_type == CNN_ATTENTION_BiLSTM_MODEL:
        return build_cnn_attention_bilstm_model(train_size, input_shape, forecast_period)
    elif model_type == CNN_ATTENTION_BiGRU_MODEL:
        return build_cnn_attention_bigru_model(train_size, input_shape, forecast_period)
    #-----------------------------Deep More Hybrid + Attention models------------------------------- 
    elif model_type == CNN_ATTENTION_LSTM_ATTENTION_MODEL:
        return build_cnn_attention_lstm_attention_model(train_size, input_shape, forecast_period)
    elif model_type == CNN_ATTENTION_GRU_ATTENTION_MODEL:
        return build_cnn_attention_gru_attention_model(train_size, input_shape, forecast_period)
    elif model_type == CNN_ATTENTION_BiLSTM_ATTENTION_MODEL:
        return build_cnn_attention_bilstm_attention_model(train_size, input_shape, forecast_period)
    elif model_type == CNN_ATTENTION_BiGRU_ATTENTION_MODEL:
        return build_cnn_attention_bigru_attention_model(train_size, input_shape, forecast_period)
    else:
        raise ValueError("Invalid model type. Please choose from the available models.")

'''-----------------------------Simple models-------------------------------'''
def build_lstm_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape))
    model.add(LSTM(200, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(200))
    model.add(Dropout(0.3))
    model.add(Dense(forecast_period))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def build_gru_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape))
    model.add(GRU(200, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(GRU(200))
    model.add(Dropout(0.3))
    model.add(Dense(forecast_period))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def build_cnn_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(32,activation='relu'))
    model.add(Dense(forecast_period))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

'''-----------------------------Simple Bi models-------------------------------'''
def build_bilstm_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape))
    model.add(Bidirectional(LSTM(200, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(200)))
    model.add(Dropout(0.3))
    model.add(Dense(forecast_period))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def build_bigru_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape))
    model.add(Bidirectional(GRU(100, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(GRU(100)))
    model.add(Dropout(0.3))
    model.add(Dense(forecast_period))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


'''-----------------------------Simple + Attention models-------------------------------'''
def build_lstm_attention_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    lookback = input_shape[0]
    inputs = Input(shape=input_shape)
    #By masking the zeros, the model can learn to ignore the missing values and focus on the valid data.
    masked = Masking(mask_value=0.)(inputs)
    lstm1 = LSTM(200, return_sequences=True)(masked)
    dropout1 = Dropout(0.3)(lstm1)
    lstm2 = LSTM(200, return_sequences=True)(dropout1)
    dropout2 = Dropout(0.3)(lstm2)
    
    attention = dot([dropout2, dropout2], axes=[2, 2])
    # extracting weight for every observation in the history size, equivalent to look back!
    attention = Dense(lookback, activation='relu')(attention)
    # assinging weight to lstm by dot product
    context = dot([attention, dropout2], axes=[2, 1])
    flattened = Flatten()(context)
    output = Dense(forecast_period)(flattened)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def build_gru_attention_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    lookback = input_shape[0]
    inputs = Input(shape=input_shape)
    #By masking the zeros, the model can learn to ignore the missing values and focus on the valid data.
    masked = Masking(mask_value=0.)(inputs)
    gru1 = GRU(200, return_sequences=True)(masked)
    dropout1 = Dropout(0.3)(gru1)
    gru2 = GRU(200, return_sequences=True)(dropout1)
    dropout2 = Dropout(0.3)(gru2)
    
    attention = dot([dropout2, dropout2], axes=[2, 2])
    # extracting weight for every observation in the history size, equivalent to look back!
    attention = Dense(lookback, activation='relu')(attention)
    # assinging weight to lstm by dot product
    context = dot([attention, dropout2], axes=[2, 1])
    flattened = Flatten()(context)
    output = Dense(forecast_period)(flattened)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def build_cnn_attention_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    lookback = input_shape[0]
    inputs = Input(shape=input_shape)
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    conv1_bn = BatchNormalization()(conv1)
    max_pooling1 = MaxPooling1D(pool_size=2)(conv1_bn)
    conv2 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(max_pooling1)
    conv2_bn = BatchNormalization()(conv2)
    max_pooling2 = MaxPooling1D(pool_size=2)(conv2_bn)
    
    attention = dot([max_pooling2, max_pooling2], axes=[2, 2])
    # Extracting weight for every observation in the history size, equivalent to look back!
    attention = Dense(lookback, activation='relu')(attention)
    # Assigning weight to CNN by dot product
    context = Concatenate(axis=-1)([max_pooling2, attention])
    flattened = Flatten()(context)
    output = Dense(forecast_period)(flattened)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


'''-----------------------------Bi + Attention models-------------------------------'''
def build_bilstm_attention_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    lookback = input_shape[0]
    inputs = Input(shape=input_shape)
    #By masking the zeros, the model can learn to ignore the missing values and focus on the valid data.
    masked = Masking(mask_value=0.)(inputs)
    bilstm1 = Bidirectional(LSTM(200, return_sequences=True))(masked)
    dropout1 = Dropout(0.3)(bilstm1)
    bilstm2 = Bidirectional(LSTM(200, return_sequences=True))(dropout1)
    dropout2 = Dropout(0.3)(bilstm2)
    
    attention = dot([dropout2, dropout2], axes=[2, 2])
    # extracting weight for every observation in the history size, equivalent to look back!
    attention = Dense(lookback, activation='relu')(attention)
    # assinging weight to lstm by dot product
    context = dot([attention, dropout2], axes=[2, 1])
    flattened = Flatten()(context)
    output = Dense(forecast_period)(flattened)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def build_bigru_attention_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    lookback = input_shape[0]
    inputs = Input(shape=input_shape)
    #By masking the zeros, the model can learn to ignore the missing values and focus on the valid data.
    masked = Masking(mask_value=0.)(inputs)
    bigru1 = Bidirectional(GRU(200, return_sequences=True))(masked)
    dropout1 = Dropout(0.3)(bigru1)
    bigru2 = Bidirectional(GRU(200, return_sequences=True))(dropout1)
    dropout2 = Dropout(0.3)(bigru2)
    
    attention = dot([dropout2, dropout2], axes=[2, 2])
    # extracting weight for every observation in the history size, equivalent to look back!
    attention = Dense(lookback, activation='relu')(attention)
    # assinging weight to lstm by dot product
    context = dot([attention, dropout2], axes=[2, 1])
    flattened = Flatten()(context)
    output = Dense(forecast_period)(flattened)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


'''-----------------------------Hybrid models-------------------------------'''

def build_cnn_lstm_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(LSTM(200, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(200))
    model.add(Dropout(0.3))
    
    model.add(Dense(forecast_period))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def build_cnn_gru_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(GRU(200, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(GRU(200))
    model.add(Dropout(0.3))
    
    model.add(Dense(forecast_period))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def build_cnn_bilstm_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Bidirectional(LSTM(200, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(200)))
    model.add(Dropout(0.3))
    
    model.add(Dense(forecast_period))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def build_cnn_bigru_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Bidirectional(GRU(200, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(GRU(200)))
    model.add(Dropout(0.3))
    
    model.add(Dense(forecast_period))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


'''-----------------------------Hybrid + Attention models-------------------------------'''
def build_cnn_lstm_attention_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    lookback = input_shape[0]
    
    inputs = Input(shape=input_shape)
    masked_inputs = Masking(mask_value=0.)(inputs)
    
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(masked_inputs)
    conv1_bn = BatchNormalization()(conv1)
    max_pooling1 = MaxPooling1D(pool_size=2)(conv1_bn)
    conv2 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(max_pooling1)
    conv2_bn = BatchNormalization()(conv2)
    max_pooling2 = MaxPooling1D(pool_size=2)(conv2_bn)
    
    lstm1 = LSTM(200, return_sequences=True)(max_pooling2)
    dropout1 = Dropout(0.3)(lstm1)
    lstm2 = LSTM(200, return_sequences=True)(dropout1)
    dropout2 = Dropout(0.3)(lstm2)
    
    attention = dot([dropout2, dropout2], axes=[2, 2])
    attention = Dense(lookback, activation='relu')(attention)
    context = Concatenate(axis=-1)([max_pooling2, dropout2])
    flattened = Flatten()(context)
    output = Dense(forecast_period)(flattened)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def build_cnn_gru_attention_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    lookback = input_shape[0]
    
    inputs = Input(shape=input_shape)
    masked_inputs = Masking(mask_value=0.)(inputs)
    
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(masked_inputs)
    conv1_bn = BatchNormalization()(conv1)
    max_pooling1 = MaxPooling1D(pool_size=2)(conv1_bn)
    conv2 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(max_pooling1)
    conv2_bn = BatchNormalization()(conv2)
    max_pooling2 = MaxPooling1D(pool_size=2)(conv2_bn)
    
    gru1 = GRU(200, return_sequences=True)(max_pooling2)
    dropout1 = Dropout(0.3)(gru1)
    gru2 = GRU(200, return_sequences=True)(dropout1)
    dropout2 = Dropout(0.3)(gru2)
    
    attention = dot([dropout2, dropout2], axes=[2, 2])
    attention = Dense(lookback, activation='relu')(attention)
    context = Concatenate(axis=-1)([max_pooling2, dropout2])
    flattened = Flatten()(context)
    output = Dense(forecast_period)(flattened)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def build_cnn_bilstm_attention_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    lookback = input_shape[0]
    
    inputs = Input(shape=input_shape)
    masked_inputs = Masking(mask_value=0.)(inputs)
    
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(masked_inputs)
    conv1_bn = BatchNormalization()(conv1)
    max_pooling1 = MaxPooling1D(pool_size=2)(conv1_bn)
    conv2 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(max_pooling1)
    conv2_bn = BatchNormalization()(conv2)
    max_pooling2 = MaxPooling1D(pool_size=2)(conv2_bn)
    
    bilstm1 = Bidirectional(LSTM(200, return_sequences=True))(max_pooling2)
    dropout1 = Dropout(0.3)(bilstm1)
    bilstm2 = Bidirectional(LSTM(200, return_sequences=True))(dropout1)
    dropout2 = Dropout(0.3)(bilstm2)
    
    attention = dot([dropout2, dropout2], axes=[2, 2])
    attention = Dense(lookback, activation='relu')(attention)
    context = Concatenate(axis=-1)([max_pooling2, dropout2])
    flattened = Flatten()(context)
    output = Dense(forecast_period)(flattened)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def build_cnn_bigru_attention_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    lookback = input_shape[0]
    
    inputs = Input(shape=input_shape)
    masked_inputs = Masking(mask_value=0.)(inputs)
    
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(masked_inputs)
    conv1_bn = BatchNormalization()(conv1)
    max_pooling1 = MaxPooling1D(pool_size=2)(conv1_bn)
    conv2 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(max_pooling1)
    conv2_bn = BatchNormalization()(conv2)
    max_pooling2 = MaxPooling1D(pool_size=2)(conv2_bn)
    
    bigru1 = Bidirectional(GRU(200, return_sequences=True))(max_pooling2)
    dropout1 = Dropout(0.3)(bigru1)
    bigru2 = Bidirectional(GRU(200, return_sequences=True))(dropout1)
    dropout2 = Dropout(0.3)(bigru2)
    
    attention = dot([dropout2, dropout2], axes=[2, 2])
    attention = Dense(lookback, activation='relu')(attention)
    context = Concatenate(axis=-1)([max_pooling2, dropout2])
    flattened = Flatten()(context)
    output = Dense(forecast_period)(flattened)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


'''-----------------------------Deep Hybrid + Attention models-------------------------------'''    
def build_cnn_attention_lstm_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    inputs = Input(shape=input_shape)
    masked_inputs = Masking(mask_value=0.)(inputs)
    
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(masked_inputs)
    conv1_bn = BatchNormalization()(conv1)
    max_pooling1 = MaxPooling1D(pool_size=2)(conv1_bn)
    conv2 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(max_pooling1)
    conv2_bn = BatchNormalization()(conv2)
    max_pooling2 = MaxPooling1D(pool_size=2)(conv2_bn)
    
    lstm1 = LSTM(200, return_sequences=True)(max_pooling2)
    dropout1 = Dropout(0.3)(lstm1)
    lstm2 = LSTM(200, return_sequences=True)(dropout1)
    dropout2 = Dropout(0.3)(lstm2)
    
    attention = Concatenate(axis=-1)([dropout2, max_pooling2])
    attention = Dense(input_shape[0], activation='relu')(attention)
    attention = Permute((2, 1))(attention)
    context = dot([attention, dropout2], axes=[2, 1])
    
    flattened = Flatten()(context)
    output = Dense(forecast_period)(flattened)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def build_cnn_attention_gru_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    inputs = Input(shape=input_shape)
    masked_inputs = Masking(mask_value=0.)(inputs)
    
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(masked_inputs)
    conv1_bn = BatchNormalization()(conv1)
    max_pooling1 = MaxPooling1D(pool_size=2)(conv1_bn)
    conv2 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(max_pooling1)
    conv2_bn = BatchNormalization()(conv2)
    max_pooling2 = MaxPooling1D(pool_size=2)(conv2_bn)
    
    gru1 = GRU(200, return_sequences=True)(max_pooling2)
    dropout1 = Dropout(0.3)(gru1)
    gru2 = GRU(200, return_sequences=True)(dropout1)
    dropout2 = Dropout(0.3)(gru2)
    
    attention = Concatenate(axis=-1)([dropout2, max_pooling2])
    attention = Dense(input_shape[0], activation='relu')(attention)
    attention = Permute((2, 1))(attention)
    context = dot([attention, dropout2], axes=[2, 1])
    
    flattened = Flatten()(context)
    output = Dense(forecast_period)(flattened)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def build_cnn_attention_bilstm_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    inputs = Input(shape=input_shape)
    masked_inputs = Masking(mask_value=0.)(inputs)
    
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(masked_inputs)
    conv1_bn = BatchNormalization()(conv1)
    max_pooling1 = MaxPooling1D(pool_size=2)(conv1_bn)
    conv2 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(max_pooling1)
    conv2_bn = BatchNormalization()(conv2)
    max_pooling2 = MaxPooling1D(pool_size=2)(conv2_bn)
    
    bilstm1 = Bidirectional(LSTM(200, return_sequences=True))(max_pooling2)
    dropout1 = Dropout(0.3)(bilstm1)
    bilstm2 = Bidirectional(LSTM(200, return_sequences=True))(dropout1)
    dropout2 = Dropout(0.3)(bilstm2)
    
    attention = Concatenate(axis=-1)([dropout2, max_pooling2])
    attention = Dense(input_shape[0], activation='relu')(attention)
    attention = Permute((2, 1))(attention)
    context = dot([attention, dropout2], axes=[2, 1])
    
    flattened = Flatten()(context)
    output = Dense(forecast_period)(flattened)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def build_cnn_attention_bigru_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    inputs = Input(shape=input_shape)
    masked_inputs = Masking(mask_value=0.)(inputs)
    
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(masked_inputs)
    conv1_bn = BatchNormalization()(conv1)
    max_pooling1 = MaxPooling1D(pool_size=2)(conv1_bn)
    conv2 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(max_pooling1)
    conv2_bn = BatchNormalization()(conv2)
    max_pooling2 = MaxPooling1D(pool_size=2)(conv2_bn)
    
    bigru1 = Bidirectional(GRU(200, return_sequences=True))(max_pooling2)
    dropout1 = Dropout(0.3)(bigru1)
    bigru2 = Bidirectional(GRU(200, return_sequences=True))(dropout1)
    dropout2 = Dropout(0.3)(bigru2)
    
    attention = Concatenate(axis=-1)([dropout2, max_pooling2])
    attention = Dense(input_shape[0], activation='relu')(attention)
    attention = Permute((2, 1))(attention)
    context = dot([attention, dropout2], axes=[2, 1])
    
    flattened = Flatten()(context)
    output = Dense(forecast_period)(flattened)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


'''-----------------------------Deep More Hybrid + Attention models-------------------------------'''    
def build_cnn_attention_lstm_attention_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    inputs = Input(shape=input_shape)
    masked_inputs = Masking(mask_value=0.)(inputs)
    
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(masked_inputs)
    conv1_bn = BatchNormalization()(conv1)
    max_pooling1 = MaxPooling1D(pool_size=2)(conv1_bn)
    conv2 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(max_pooling1)
    conv2_bn = BatchNormalization()(conv2)
    max_pooling2 = MaxPooling1D(pool_size=2)(conv2_bn)
    
    lstm1 = LSTM(200, return_sequences=True)(max_pooling2)
    dropout1 = Dropout(0.3)(lstm1)
    lstm2 = LSTM(200, return_sequences=True)(dropout1)
    dropout2 = Dropout(0.3)(lstm2)
    
    # First Attention Mechanism
    attention1 = Dense(input_shape[0], activation='relu')(dropout2)
    attention1 = Permute((2, 1))(attention1)
    context1 = dot([attention1, dropout2], axes=[2, 1])
    
    # Second Attention Mechanism
    attention2 = Dense(input_shape[0], activation='relu')(context1)
    attention2 = Permute((2, 1))(attention2)
    context2 = dot([attention2, context1], axes=[2, 1])
    
    flattened = Flatten()(context2)
    output = Dense(forecast_period)(flattened)
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    
    return model

def build_cnn_attention_gru_attention_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    inputs = Input(shape=input_shape)
    masked_inputs = Masking(mask_value=0.)(inputs)
    
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(masked_inputs)
    conv1_bn = BatchNormalization()(conv1)
    max_pooling1 = MaxPooling1D(pool_size=2)(conv1_bn)
    conv2 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(max_pooling1)
    conv2_bn = BatchNormalization()(conv2)
    max_pooling2 = MaxPooling1D(pool_size=2)(conv2_bn)
    
    gru1 = GRU(200, return_sequences=True)(max_pooling2)
    dropout1 = Dropout(0.3)(gru1)
    gru2 = GRU(200, return_sequences=True)(dropout1)
    dropout2 = Dropout(0.3)(gru2)
    
    # First Attention Mechanism
    attention1 = Dense(input_shape[0], activation='relu')(dropout2)
    attention1 = Permute((2, 1))(attention1)
    context1 = dot([attention1, dropout2], axes=[2, 1])
    
    # Second Attention Mechanism
    attention2 = Dense(input_shape[0], activation='relu')(context1)
    attention2 = Permute((2, 1))(attention2)
    context2 = dot([attention2, context1], axes=[2, 1])
    
    flattened = Flatten()(context2)
    output = Dense(forecast_period)(flattened)
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    
    return model

def build_cnn_attention_bilstm_attention_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    inputs = Input(shape=input_shape)
    masked_inputs = Masking(mask_value=0.)(inputs)
    
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(masked_inputs)
    conv1_bn = BatchNormalization()(conv1)
    max_pooling1 = MaxPooling1D(pool_size=2)(conv1_bn)
    conv2 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(max_pooling1)
    conv2_bn = BatchNormalization()(conv2)
    max_pooling2 = MaxPooling1D(pool_size=2)(conv2_bn)
    
    bilstm1 = Bidirectional(LSTM(200, return_sequences=True))(max_pooling2)
    dropout1 = Dropout(0.3)(bilstm1)
    bilstm2 = Bidirectional(LSTM(200, return_sequences=True))(dropout1)
    dropout2 = Dropout(0.3)(bilstm2)
    
    # First Attention Mechanism
    attention1 = Dense(input_shape[0], activation='relu')(dropout2)
    attention1 = Permute((2, 1))(attention1)
    context1 = dot([attention1, dropout2], axes=[2, 1])
    
    # Second Attention Mechanism
    attention2 = Dense(input_shape[0], activation='relu')(context1)
    attention2 = Permute((2, 1))(attention2)
    context2 = dot([attention2, context1], axes=[2, 1])
    
    flattened = Flatten()(context2)
    output = Dense(forecast_period)(flattened)
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    
    return model

def build_cnn_attention_bigru_attention_model(train_size, input_shape, forecast_period):
    optimizer = custom_optimizer(train_size=train_size)
    
    inputs = Input(shape=input_shape)
    masked_inputs = Masking(mask_value=0.)(inputs)
    
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(masked_inputs)
    conv1_bn = BatchNormalization()(conv1)
    max_pooling1 = MaxPooling1D(pool_size=2)(conv1_bn)
    conv2 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(max_pooling1)
    conv2_bn = BatchNormalization()(conv2)
    max_pooling2 = MaxPooling1D(pool_size=2)(conv2_bn)
    
    bigru1 = Bidirectional(GRU(200, return_sequences=True))(max_pooling2)
    dropout1 = Dropout(0.3)(bigru1)
    bigru2 = Bidirectional(GRU(200, return_sequences=True))(dropout1)
    dropout2 = Dropout(0.3)(bigru2)
    
    # First Attention Mechanism
    attention1 = Dense(input_shape[0], activation='relu')(dropout2)
    attention1 = Permute((2, 1))(attention1)
    context1 = dot([attention1, dropout2], axes=[2, 1])
    
    # Second Attention Mechanism
    attention2 = Dense(input_shape[0], activation='relu')(context1)
    attention2 = Permute((2, 1))(attention2)
    context2 = dot([attention2, context1], axes=[2, 1])
    
    flattened = Flatten()(context2)
    output = Dense(forecast_period)(flattened)
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    
    return model
