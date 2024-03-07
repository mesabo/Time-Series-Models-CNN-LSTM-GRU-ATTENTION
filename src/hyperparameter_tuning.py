#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 17:02:22 2024

@author: mesabo
"""

from sklearn.model_selection import GridSearchCV

from keras.models import Sequential, Model
from keras.layers import (LSTM, Dense, Flatten, Conv1D, MaxPooling1D, GRU, 
                          Bidirectional, TimeDistributed, Attention, Input,
                          Reshape, RepeatVector,Masking, Concatenate, dot,
                          Permute, Dropout, BatchNormalization)
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
from keras.optimizers import Adam, SGD, RMSprop
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

from kerastuner import Hyperband
from kerastuner.engine.hyperparameters import HyperParameters

from tensorflow.keras.optimizers.schedules import ExponentialDecay
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
    EPOCH, BATCH_SIZE, CHECK_PATH, PARAMS_GRID, CHECK_HYPERBAND
)



def custom_optimizer(hp_learning_rate, decay_steps, hp_optimizer_choice):
    decay_rate = 0.9
    lr_schedule = ExponentialDecay(
        initial_learning_rate=hp_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate
    )

    if hp_optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=lr_schedule, amsgrad=True)
    elif hp_optimizer_choice == 'sgd':
        optimizer = SGD(learning_rate=lr_schedule, momentum=0.9)
    elif hp_optimizer_choice == 'rmsprop':
        optimizer = RMSprop(learning_rate=lr_schedule)
    else:
        raise ValueError("Unknown optimizer type. Supported types are 'adam', 'sgd', and 'rmsprop'.")

    return optimizer


'''-----------------------------Simple models-------------------------------'''
def build_lstm_model(hp, input_shape, forecast_period):
    hp_units = hp.Int('units', min_value=50, max_value=300, step=50)
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    decay_steps = hp.Int('decay_steps', min_value=5000, max_value=20000, step=5000)
    hp_optimizer_choice = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])
    hp_num_layers = hp.Int('num_layers', min_value=1, max_value=3, step=1)  # Define number of layers

    optimizer = custom_optimizer(hp_learning_rate, decay_steps, hp_optimizer_choice)

    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape))
    model.add(LSTM(units=hp_units, return_sequences=True, activation='relu'))
    model.add(Dropout(rate=hp_dropout))
    model.add(LSTM(units=hp_units, activation='relu'))
    model.add(Dropout(rate=hp_dropout))
    model.add(Dense(units=forecast_period, activation='softmax'))
    model.compile(loss='mean_absolute_error', optimizer=optimizer)
    model.summary()
    return model


def build_gru_model(hp, input_shape, forecast_period):
    hp_units = hp.Int('units', min_value=50, max_value=300, step=50)
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    decay_steps = hp.Int('decay_steps', min_value=5000, max_value=20000, step=5000)
    hp_optimizer_choice = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])
    hp_num_layers = hp.Int('num_layers', min_value=1, max_value=3, step=1)  # Define number of layers

    optimizer = custom_optimizer(hp_learning_rate, decay_steps, hp_optimizer_choice)

    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape))

    # Add LSTM layers based on the value of hp_num_layers
    for i in range(hp_num_layers):
        if i == 0:
            model.add(GRU(units=hp_units, return_sequences=True, activation='relu'))
        else:
            model.add(GRU(units=hp_units, return_sequences=(i < hp_num_layers - 1), activation='relu'))
        model.add(Dropout(rate=hp_dropout))

    model.add(Dense(units=forecast_period, activation='softmax'))
    model.compile(loss='mean_absolute_error', optimizer=optimizer)
    model.summary()
    return model

def build_cnn_model(hp, input_shape, forecast_period):
    hp_filters = hp.Choice('filters', values=[32, 64, 128])
    hp_kernel_size = hp.Choice('kernel_size', values=[1, 2, 3, 5])
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    decay_steps = hp.Int('decay_steps', min_value=5000, max_value=25000, step=5000)
    hp_optimizer_choice = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])

    optimizer = custom_optimizer(hp_learning_rate, decay_steps, hp_optimizer_choice)

    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape))
    model.add(Conv1D(filters=hp_filters, kernel_size=hp_kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=hp_filters*2, kernel_size=hp_kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(forecast_period))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.summary()
    return model

'''-----------------------------Simple Bi models-------------------------------'''



def build_model(hp, model_type, input_shape, forecast_period):
    #-----------------------------Simple models-------------------------------
    if model_type == LSTM_MODEL:
        return build_lstm_model(hp, input_shape, forecast_period)
    elif model_type == GRU_MODEL:
        return build_gru_model(hp, input_shape, forecast_period)
    elif model_type == CNN_MODEL:
        return build_cnn_model(hp, input_shape, forecast_period)
    else:
        raise ValueError("Invalid model type. Please choose from the available models.")
    '''
    #-----------------------------Simple Bi models-------------------------------
    elif model_type == BiLSTM_MODEL:
        return build_bilstm_model(hp, input_shape, forecast_period)
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
    '''



def tune_custom_model(dataX, input_shape, forecast_period, model_type):
    x_train, y_train, x_val, y_val = dataX
    
    def build_hypermodel(hp):
        return build_model(hp, model_type, input_shape, forecast_period)

    tuner = Hyperband(
        build_hypermodel,
        objective='val_loss',
        max_epochs=50,
        factor=3,
        hyperband_iterations=3,
        directory=CHECK_HYPERBAND,
        project_name=model_type
    )

    #tuner.search_space_summary()
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=1)

    tuner.search(x_train, y_train, epochs=20, validation_data=(x_val, y_val), callbacks=[early_stopping])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    return best_hps
