#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 17:02:22 2024

@author: mesabo
"""
from math import ceil
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential, Model
from keras.layers import (LSTM, Dense, Flatten, Conv1D, MaxPooling1D, GRU,
                          Bidirectional, TimeDistributed, Attention, Input,
                          Reshape, RepeatVector, Masking, Concatenate, dot,
                          Permute, Dropout, BatchNormalization)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam, SGD, RMSprop
from scikeras.wrappers import KerasClassifier, KerasRegressor
from keras.regularizers import l1, l2, l1_l2

from kerastuner import Hyperband
from kerastuner.engine.hyperparameters import HyperParameters

from tensorflow.keras.optimizers.schedules import ExponentialDecay
import tensorflow as tf

from constants import (
    LSTM_MODEL, GRU_MODEL, CNN_MODEL, BiLSTM_MODEL, BiGRU_MODEL,
    LSTM_ATTENTION_MODEL, GRU_ATTENTION_MODEL, CNN_ATTENTION_MODEL,
    BiLSTM_ATTENTION_MODEL, BiGRU_ATTENTION_MODEL,
    CNN_LSTM_MODEL, CNN_GRU_MODEL, CNN_BiLSTM_MODEL, CNN_BiGRU_MODEL,
    CNN_LSTM_ATTENTION_MODEL, CNN_GRU_ATTENTION_MODEL,
    CNN_BiLSTM_ATTENTION_MODEL, CNN_BiGRU_ATTENTION_MODEL,
    CNN_ATTENTION_LSTM_ATTENTION_MODEL, CNN_ATTENTION_GRU_ATTENTION_MODEL,
    CNN_ATTENTION_BiLSTM_ATTENTION_MODEL, CNN_ATTENTION_BiGRU_ATTENTION_MODEL,
    CNN_ATTENTION_LSTM_MODEL, CNN_ATTENTION_GRU_MODEL,
    CNN_ATTENTION_BiLSTM_MODEL, CNN_ATTENTION_BiGRU_MODEL,
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
        raise ValueError(
            "Unknown optimizer type. Supported types are 'adam', 'sgd', and 'rmsprop'.")

    return optimizer


'''-----------------------------Simple models-------------------------------'''


def build_lstm_model(hp, input_shape, forecast_period):
    hp_units = hp.Int('units', min_value=50, max_value=200, step=50)
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    decay_steps = hp.Int('decay_steps', min_value=5000,
                         max_value=20000, step=5000)
    hp_optimizer_choice = hp.Choice(
        'optimizer', values=['adam', 'sgd', 'rmsprop'])
    hp_l1_regularizer = hp.Choice('l1_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_l2_regularizer = hp.Choice('l2_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_activation = hp.Choice('activation', values=['linear', 'relu', 'softmax'])

    num_layers = hp.Int('num_layers', min_value=1, max_value=3, step=1)
    lstm_layers = []

    for i in range(num_layers):
        units = hp.Int(f'lstm_units_{i}', min_value=50,
                       max_value=200, step=50, default=hp_units)
        dropout = hp.Float(
            f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1, default=hp_dropout)
        return_sequences = i < num_layers - 1

        lstm_layers.append(
            LSTM(units=units, return_sequences=return_sequences, activation='relu'))
        lstm_layers.append(Dropout(rate=dropout))

    optimizer = custom_optimizer(
        hp_learning_rate, decay_steps, hp_optimizer_choice)

    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape))

    for layer in lstm_layers:
        model.add(layer)

    model.add(Dense(units=forecast_period, activation=hp_activation,
                    kernel_regularizer=l1(hp_l1_regularizer),
                    activity_regularizer=l2(hp_l2_regularizer)))

    model.compile(loss='mean_absolute_error', optimizer=optimizer)
    model.summary()
    return model


def build_gru_model(hp, input_shape, forecast_period):
    hp_units = hp.Int('units', min_value=50, max_value=200, step=50)
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    decay_steps = hp.Int('decay_steps', min_value=5000,
                         max_value=20000, step=5000)
    hp_optimizer_choice = hp.Choice(
        'optimizer', values=['adam', 'sgd', 'rmsprop'])
    hp_l1_regularizer = hp.Choice('l1_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_l2_regularizer = hp.Choice('l2_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_activation = hp.Choice('activation', values=['linear', 'relu', 'softmax'])

    num_layers = hp.Int('num_layers', min_value=1, max_value=3, step=1)
    gru_layers = []

    for i in range(num_layers):
        units = hp.Int(f'gru_units_{i}', min_value=50,
                       max_value=200, step=50, default=hp_units)
        dropout = hp.Float(
            f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1, default=hp_dropout)
        return_sequences = i < num_layers - 1

        gru_layers.append(
            GRU(units=units, return_sequences=return_sequences, activation='relu'))
        gru_layers.append(Dropout(rate=dropout))

    optimizer = custom_optimizer(
        hp_learning_rate, decay_steps, hp_optimizer_choice)

    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape))

    for layer in gru_layers:
        model.add(layer)

    model.add(Dense(units=forecast_period, activation=hp_activation,
                    kernel_regularizer=l1(hp_l1_regularizer),
                    activity_regularizer=l2(hp_l2_regularizer)))

    model.compile(loss='mean_absolute_error', optimizer=optimizer)
    model.summary()
    return model


def build_cnn_model(hp, input_shape, forecast_period):
    hp_num_layers = hp.Int('num_layers', min_value=1, max_value=3, step=1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    decay_steps = hp.Int('decay_steps', min_value=5000,
                         max_value=25000, step=5000)
    hp_optimizer_choice = hp.Choice(
        'optimizer', values=['adam', 'sgd', 'rmsprop'])
    hp_dropout_rate = hp.Float(
        'dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    hp_l1_regularizer = hp.Choice('l1_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_l2_regularizer = hp.Choice('l2_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_activation = hp.Choice('activation', values=['linear', 'relu', 'softmax'])

    optimizer = custom_optimizer(
        hp_learning_rate, decay_steps, hp_optimizer_choice)

    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape))

    for i in range(hp_num_layers):
        filters = hp.Choice(f'filters_{i}', values=[32, 64, 96, 128])
        kernel_size = hp.Choice(f'kernel_size_{i}', values=[1, 2, 3, 5])

        model.add(Conv1D(filters=filters, kernel_size=kernel_size,
                         activation='relu', padding='same'))
        model.add(BatchNormalization())

        pool_size = ceil(model.layers[-1].output_shape[1] / 2)
        if pool_size > 1:
            model.add(MaxPooling1D(pool_size=pool_size))
        else:
            model.add(MaxPooling1D(pool_size=1))

        model.add(Dropout(rate=hp_dropout_rate))

    model.add(Flatten())
    model.add(Dense(32, activation='relu', kernel_regularizer=l1(hp_l1_regularizer),
                    activity_regularizer=l2(hp_l2_regularizer)))
    model.add(Dense(forecast_period, activation=hp_activation))
    model.compile(loss='mean_absolute_error', optimizer=optimizer)
    model.summary()
    return model


'''-----------------------------Simple Bi models-------------------------------'''


def build_bilstm_model(hp, input_shape, forecast_period):
    hp_units = hp.Int('units', min_value=50, max_value=200, step=50)
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    decay_steps = hp.Int('decay_steps', min_value=5000,
                         max_value=20000, step=5000)
    hp_optimizer_choice = hp.Choice(
        'optimizer', values=['adam', 'sgd', 'rmsprop'])
    hp_l1_regularizer = hp.Choice('l1_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_l2_regularizer = hp.Choice('l2_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_activation = hp.Choice('activation', values=['linear', 'relu', 'softmax'])

    num_layers = hp.Int('num_layers', min_value=1, max_value=3, step=1)
    lstm_layers = []

    for i in range(num_layers):
        units = hp.Int(f'bilstm_units_{i}', min_value=50,
                       max_value=200, step=50, default=hp_units)
        dropout = hp.Float(
            f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1, default=hp_dropout)
        return_sequences = i < num_layers - 1

        lstm_layers.append(Bidirectional(
            LSTM(units=units, return_sequences=return_sequences, activation='relu')))
        lstm_layers.append(Dropout(rate=dropout))

    optimizer = custom_optimizer(
        hp_learning_rate, decay_steps, hp_optimizer_choice)

    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape))

    for layer in lstm_layers:
        model.add(layer)

    model.add(Dense(units=forecast_period, activation=hp_activation,
                    kernel_regularizer=l1(hp_l1_regularizer),
                    activity_regularizer=l2(hp_l2_regularizer)))

    model.compile(loss='mean_absolute_error', optimizer=optimizer)
    model.summary()
    return model


def build_bigru_model(hp, input_shape, forecast_period):
    hp_units = hp.Int('units', min_value=50, max_value=200, step=50)
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    decay_steps = hp.Int('decay_steps', min_value=5000,
                         max_value=20000, step=5000)
    hp_optimizer_choice = hp.Choice(
        'optimizer', values=['adam', 'sgd', 'rmsprop'])
    hp_l1_regularizer = hp.Choice('l1_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_l2_regularizer = hp.Choice('l2_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_activation = hp.Choice('activation', values=['linear', 'relu', 'softmax'])

    num_layers = hp.Int('num_layers', min_value=1, max_value=3, step=1)
    bigru_layers = []

    for i in range(num_layers):
        units = hp.Int(f'bigru_units_{i}', min_value=50,
                       max_value=200, step=50, default=hp_units)
        dropout = hp.Float(
            f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1, default=hp_dropout)
        return_sequences = i < num_layers - 1

        bigru_layers.append(Bidirectional(
            GRU(units=units, return_sequences=return_sequences, activation='relu')))
        bigru_layers.append(Dropout(rate=dropout))

    optimizer = custom_optimizer(
        hp_learning_rate, decay_steps, hp_optimizer_choice)

    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape))

    for layer in bigru_layers:
        model.add(layer)

    model.add(Dense(units=forecast_period, activation=hp_activation,
                    kernel_regularizer=l1(hp_l1_regularizer),
                    activity_regularizer=l2(hp_l2_regularizer)))

    model.compile(loss='mean_absolute_error', optimizer=optimizer)
    model.summary()
    return model


'''-----------------------------Simple + Attention models-------------------------------'''


def build_lstm_attention_model(hp, input_shape, forecast_period):
    num_layers = hp.Int('num_layers', min_value=1, max_value=3, step=1)
    hp_units = hp.Int('units', min_value=50, max_value=200, step=50)
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    decay_steps = hp.Int('decay_steps', min_value=5000,
                         max_value=20000, step=5000)
    hp_optimizer_choice = hp.Choice(
        'optimizer', values=['adam', 'sgd', 'rmsprop'])
    hp_l1_regularizer = hp.Choice('l1_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_l2_regularizer = hp.Choice('l2_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_activation = hp.Choice('activation', values=['linear', 'relu', 'softmax'])

    optimizer = custom_optimizer(
        hp_learning_rate, decay_steps, hp_optimizer_choice)

    inputs = Input(shape=input_shape)
    masked = Masking(mask_value=0.)(inputs)

    lstm_layers = []
    for i in range(num_layers):
        units = hp.Int(f'lstm_units_{i}', min_value=50,
                       max_value=200, step=50, default=hp_units)
        dropout = hp.Float(
            f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1, default=hp_dropout)

        lstm_layers.append(
            LSTM(units=units, return_sequences=True, activation='relu'))
        lstm_layers.append(Dropout(rate=dropout))

    lstm_output = masked
    for layer in lstm_layers:
        lstm_output = layer(lstm_output)

    attention = dot([lstm_output, lstm_output], axes=[2, 2])
    attention = Dense(input_shape[0], activation='relu')(attention)
    context = dot([attention, lstm_output], axes=[2, 1])
    flattened = Flatten()(context)

    output = Dense(forecast_period, activation=hp_activation, kernel_regularizer=l1_l2(
        hp_l1_regularizer, hp_l2_regularizer))(flattened)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.summary()
    return model


def build_gru_attention_model(hp, input_shape, forecast_period):
    num_layers = hp.Int('num_layers', min_value=1, max_value=3, step=1)
    hp_units = hp.Int('units', min_value=50, max_value=200, step=50)
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    decay_steps = hp.Int('decay_steps', min_value=5000,
                         max_value=20000, step=5000)
    hp_optimizer_choice = hp.Choice(
        'optimizer', values=['adam', 'sgd', 'rmsprop'])
    hp_l1_regularizer = hp.Choice('l1_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_l2_regularizer = hp.Choice('l2_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_activation = hp.Choice('activation', values=['linear', 'relu', 'softmax'])

    optimizer = custom_optimizer(
        hp_learning_rate, decay_steps, hp_optimizer_choice)

    inputs = Input(shape=input_shape)
    masked = Masking(mask_value=0.)(inputs)

    gru_layers = []
    for i in range(num_layers):
        units = hp.Int(f'gru_units_{i}', min_value=50,
                       max_value=200, step=50, default=hp_units)
        dropout = hp.Float(
            f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1, default=hp_dropout)

        gru_layers.append(
            LSTM(units=units, return_sequences=True, activation='relu'))
        gru_layers.append(Dropout(rate=dropout))

    gru_output = masked
    for layer in gru_layers:
        gru_output = layer(gru_output)

    attention = dot([gru_output, gru_output], axes=[2, 2])
    attention = Dense(input_shape[0], activation='relu')(attention)
    context = dot([attention, gru_output], axes=[2, 1])
    flattened = Flatten()(context)

    output = Dense(forecast_period, activation=hp_activation, kernel_regularizer=l1_l2(
        hp_l1_regularizer, hp_l2_regularizer))(flattened)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.summary()
    return model


def build_cnn_attention_model(hp, input_shape, forecast_period):
    hp_num_layers = hp.Int('num_layers', min_value=1, max_value=3, step=1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    decay_steps = hp.Int('decay_steps', min_value=5000,
                         max_value=25000, step=5000)
    hp_optimizer_choice = hp.Choice(
        'optimizer', values=['adam', 'sgd', 'rmsprop'])
    hp_l1_regularizer = hp.Choice('l1_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_l2_regularizer = hp.Choice('l2_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_activation = hp.Choice('activation', values=['linear', 'relu', 'softmax'])
    hp_dropout_rate = hp.Float(
        'dropout_rate', min_value=0.1, max_value=0.5, step=0.1)

    optimizer = custom_optimizer(
        hp_learning_rate, decay_steps, hp_optimizer_choice)

    inputs = Input(shape=input_shape)
    x = inputs

    for i in range(hp_num_layers):
        filters = hp.Choice(f'filters_{i}', values=[32, 64, 96, 128])
        kernel_size = hp.Choice(f'kernel_size_{i}', values=[1, 2, 3, 5])

        x = Conv1D(filters=filters, kernel_size=kernel_size,
                   activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        pool_size = ceil(x.shape[1] / 2)
        if pool_size > 1:
            x = MaxPooling1D(pool_size=pool_size)(x)
        else:
            x = MaxPooling1D(pool_size=1)(x)

        x = Dropout(rate=hp_dropout_rate)(x)

    attention = dot([x, x], axes=[2, 2])
    attention = Dense(input_shape[0], activation='relu')(attention)
    context = Concatenate(axis=-1)([x, attention])

    flattened = Flatten()(context)
    output = Dense(forecast_period, activation=hp_activation, kernel_regularizer=l1_l2(
        hp_l1_regularizer, hp_l2_regularizer))(flattened)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.summary()
    return model


'''-----------------------------Bi + Attention models-------------------------------'''


def build_bilstm_attention_model(hp, input_shape, forecast_period):
    num_layers = hp.Int('num_layers', min_value=1, max_value=3, step=1)
    hp_units = hp.Int('units', min_value=50, max_value=200, step=50)
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    decay_steps = hp.Int('decay_steps', min_value=5000,
                         max_value=20000, step=5000)
    hp_optimizer_choice = hp.Choice(
        'optimizer', values=['adam', 'sgd', 'rmsprop'])
    hp_l1_regularizer = hp.Choice('l1_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_l2_regularizer = hp.Choice('l2_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_activation = hp.Choice('activation', values=['linear', 'relu', 'softmax'])

    optimizer = custom_optimizer(
        hp_learning_rate, decay_steps, hp_optimizer_choice)

    inputs = Input(shape=input_shape)
    masked = Masking(mask_value=0.)(inputs)

    bilstm_layers = []
    for i in range(num_layers):
        units = hp.Int(f'bilstm_units_{i}', min_value=50,
                       max_value=200, step=50, default=hp_units)
        dropout = hp.Float(
            f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1, default=hp_dropout)

        bilstm_layers.append(
            Bidirectional(LSTM(units=units, return_sequences=True, activation='relu')))
        bilstm_layers.append(Dropout(rate=dropout))

    bilstm_output = masked
    for layer in bilstm_layers:
        bilstm_output = layer(bilstm_output)

    attention = dot([bilstm_output, bilstm_output], axes=[2, 2])
    attention = Dense(input_shape[0], activation='relu')(attention)
    context = dot([attention, bilstm_output], axes=[2, 1])
    flattened = Flatten()(context)

    output = Dense(forecast_period, activation=hp_activation, kernel_regularizer=l1_l2(
        hp_l1_regularizer, hp_l2_regularizer))(flattened)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.summary()
    return model


def build_bigru_attention_model(hp, input_shape, forecast_period):
    num_layers = hp.Int('num_layers', min_value=1, max_value=3, step=1)
    hp_units = hp.Int('units', min_value=50, max_value=200, step=50)
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    decay_steps = hp.Int('decay_steps', min_value=5000,
                         max_value=20000, step=5000)
    hp_optimizer_choice = hp.Choice(
        'optimizer', values=['adam', 'sgd', 'rmsprop'])
    hp_l1_regularizer = hp.Choice('l1_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_l2_regularizer = hp.Choice('l2_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_activation = hp.Choice('activation', values=['linear', 'relu', 'softmax'])

    optimizer = custom_optimizer(
        hp_learning_rate, decay_steps, hp_optimizer_choice)

    inputs = Input(shape=input_shape)
    masked = Masking(mask_value=0.)(inputs)

    bigru_layers = []
    for i in range(num_layers):
        units = hp.Int(f'bigru_units_{i}', min_value=50,
                       max_value=200, step=50, default=hp_units)
        dropout = hp.Float(
            f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1, default=hp_dropout)

        bigru_layers.append(
            Bidirectional(LSTM(units=units, return_sequences=True, activation='relu')))
        bigru_layers.append(Dropout(rate=dropout))

    bigru_output = masked
    for layer in bigru_layers:
        bigru_output = layer(bigru_output)

    attention = dot([bigru_output, bigru_output], axes=[2, 2])
    attention = Dense(input_shape[0], activation='relu')(attention)
    context = dot([attention, bigru_output], axes=[2, 1])
    flattened = Flatten()(context)

    output = Dense(forecast_period, activation=hp_activation, kernel_regularizer=l1_l2(
        hp_l1_regularizer, hp_l2_regularizer))(flattened)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.summary()
    return model


'''-----------------------------Hybrid models-------------------------------'''


def build_cnn_lstm_model(hp, input_shape, forecast_period):
    hp_num_cnn_layers = hp.Int(
        'num_cnn_layers', min_value=1, max_value=3, step=1)
    hp_num_lstm_layers = hp.Int(
        'num_lstm_layers', min_value=1, max_value=3, step=1)
    hp_filters = hp.Choice('filters', values=[32, 64, 96, 128])
    hp_kernel_size = hp.Choice('kernel_size', values=[1, 2, 3, 5])
    hp_lstm_units = hp.Int('lstm_units', min_value=50, max_value=200, step=50)
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    decay_steps = hp.Int('decay_steps', min_value=5000,
                         max_value=20000, step=5000)
    hp_optimizer_choice = hp.Choice(
        'optimizer', values=['adam', 'sgd', 'rmsprop'])
    hp_l1_regularizer = hp.Choice('l1_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_l2_regularizer = hp.Choice('l2_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_activation = hp.Choice('activation', values=['linear', 'relu', 'softmax'])

    optimizer = custom_optimizer(
        hp_learning_rate, decay_steps, hp_optimizer_choice)

    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape))

    for i in range(hp_num_cnn_layers):
        filters = hp.Choice(f'filters_{i}', values=[
            32, 64, 96, 128], default=hp_filters)
        kernel_size = hp.Choice(f'kernel_size_{i}', values=[
            1, 2, 3, 5], default=hp_kernel_size)

        model.add(Conv1D(filters=filters, kernel_size=kernel_size,
                         activation='relu', padding='same'))
        model.add(BatchNormalization())

        pool_size = ceil(model.layers[-1].output_shape[1] / 2)
        if pool_size > 1:
            model.add(MaxPooling1D(pool_size=pool_size))
        else:
            model.add(MaxPooling1D(pool_size=1))

        model.add(Dropout(rate=hp_dropout))

    for i in range(hp_num_lstm_layers):
        units = hp.Int(f'lstm_units_{i}', min_value=50,
                       max_value=200, step=50, default=hp_lstm_units)
        dropout = hp.Float(
            f'lstm_dropout_{i}', min_value=0.1, max_value=0.5, step=0.1, default=hp_dropout)
        return_sequences = i < hp_num_lstm_layers - 1

        model.add(
            LSTM(units=units, return_sequences=return_sequences, activation='relu'))
        model.add(Dropout(rate=dropout))

    model.add(Dense(units=forecast_period, activation=hp_activation, kernel_regularizer=l1_l2(
        hp_l1_regularizer, hp_l2_regularizer)))
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    model.summary()
    return model


def build_cnn_gru_model(hp, input_shape, forecast_period):
    hp_num_cnn_layers = hp.Int(
        'num_cnn_layers', min_value=1, max_value=3, step=1)
    hp_num_gru_layers = hp.Int(
        'num_gru_layers', min_value=1, max_value=3, step=1)
    hp_filters = hp.Choice('filters', values=[32, 64, 96, 128])
    hp_kernel_size = hp.Choice('kernel_size', values=[1, 2, 3, 5])
    hp_gru_units = hp.Int('gru_units', min_value=50, max_value=200, step=50)
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    decay_steps = hp.Int('decay_steps', min_value=5000,
                         max_value=20000, step=5000)
    hp_optimizer_choice = hp.Choice(
        'optimizer', values=['adam', 'sgd', 'rmsprop'])
    hp_l1_regularizer = hp.Choice('l1_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_l2_regularizer = hp.Choice('l2_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_activation = hp.Choice('activation', values=['linear', 'relu', 'softmax'])

    optimizer = custom_optimizer(
        hp_learning_rate, decay_steps, hp_optimizer_choice)

    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape))

    for i in range(hp_num_cnn_layers):
        filters = hp.Choice(f'filters_{i}', values=[
            32, 64, 96, 128], default=hp_filters)
        kernel_size = hp.Choice(f'kernel_size_{i}', values=[
            1, 2, 3, 5], default=hp_kernel_size)

        model.add(Conv1D(filters=filters, kernel_size=kernel_size,
                         activation='relu', padding='same'))
        model.add(BatchNormalization())

        pool_size = ceil(model.layers[-1].output_shape[1] / 2)
        if pool_size > 1:
            model.add(MaxPooling1D(pool_size=pool_size))
        else:
            model.add(MaxPooling1D(pool_size=1))

        model.add(Dropout(rate=hp_dropout))

    for i in range(hp_num_gru_layers):
        units = hp.Int(f'gru_units_{i}', min_value=50,
                       max_value=200, step=50, default=hp_gru_units)
        dropout = hp.Float(
            f'gru_dropout_{i}', min_value=0.1, max_value=0.5, step=0.1, default=hp_dropout)
        return_sequences = i < hp_num_gru_layers - 1

        model.add(
            GRU(units=units, return_sequences=return_sequences, activation='relu'))
        model.add(Dropout(rate=dropout))

    model.add(Dense(units=forecast_period, activation=hp_activation, kernel_regularizer=l1_l2(
        hp_l1_regularizer, hp_l2_regularizer)))
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    model.summary()
    return model


def build_cnn_bilstm_model(hp, input_shape, forecast_period):
    hp_num_cnn_layers = hp.Int(
        'num_cnn_layers', min_value=1, max_value=3, step=1)
    hp_num_bilstm_layers = hp.Int(
        'num_bilstm_layers', min_value=1, max_value=3, step=1)
    hp_filters = hp.Choice('filters', values=[32, 64, 96, 128])
    hp_kernel_size = hp.Choice('kernel_size', values=[1, 2, 3, 5])
    hp_bilstm_units = hp.Int(
        'bilstm_units', min_value=50, max_value=200, step=50)
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    decay_steps = hp.Int('decay_steps', min_value=5000,
                         max_value=20000, step=5000)
    hp_optimizer_choice = hp.Choice(
        'optimizer', values=['adam', 'sgd', 'rmsprop'])
    hp_l1_regularizer = hp.Choice('l1_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_l2_regularizer = hp.Choice('l2_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_activation = hp.Choice('activation', values=['linear', 'relu', 'softmax'])

    optimizer = custom_optimizer(
        hp_learning_rate, decay_steps, hp_optimizer_choice)

    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape))

    for i in range(hp_num_cnn_layers):
        filters = hp.Choice(f'filters_{i}', values=[
            32, 64, 96, 128], default=hp_filters)
        kernel_size = hp.Choice(f'kernel_size_{i}', values=[
            1, 2, 3, 5], default=hp_kernel_size)

        model.add(Conv1D(filters=filters, kernel_size=kernel_size,
                         activation='relu', padding='same'))
        model.add(BatchNormalization())

        pool_size = ceil(model.layers[-1].output_shape[1] / 2)
        if pool_size > 1:
            model.add(MaxPooling1D(pool_size=pool_size))
        else:
            model.add(MaxPooling1D(pool_size=1))

        model.add(Dropout(rate=hp_dropout))

    for i in range(hp_num_bilstm_layers):
        units = hp.Int(f'bilstm_units_{i}', min_value=50,
                       max_value=200, step=50, default=hp_bilstm_units)
        dropout = hp.Float(
            f'bilstm_dropout_{i}', min_value=0.1, max_value=0.5, step=0.1, default=hp_dropout)
        return_sequences = i < hp_num_bilstm_layers - 1

        model.add(Bidirectional(
            LSTM(units=units, return_sequences=return_sequences, activation='relu')))
        model.add(Dropout(rate=dropout))

    model.add(Dense(units=forecast_period, activation=hp_activation, kernel_regularizer=l1_l2(
        hp_l1_regularizer, hp_l2_regularizer)))
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    model.summary()
    return model


def build_cnn_bigru_model(hp, input_shape, forecast_period):
    hp_num_cnn_layers = hp.Int(
        'num_cnn_layers', min_value=1, max_value=3, step=1)
    hp_num_bigru_layers = hp.Int(
        'num_bigru_layers', min_value=1, max_value=3, step=1)
    hp_filters = hp.Choice('filters', values=[32, 64, 96, 128])
    hp_kernel_size = hp.Choice('kernel_size', values=[1, 2, 3, 5])
    hp_bigru_units = hp.Int('bigru_units', min_value=50,
                            max_value=200, step=50)
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    decay_steps = hp.Int('decay_steps', min_value=5000,
                         max_value=20000, step=5000)
    hp_optimizer_choice = hp.Choice(
        'optimizer', values=['adam', 'sgd', 'rmsprop'])
    hp_l1_regularizer = hp.Choice('l1_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_l2_regularizer = hp.Choice('l2_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_activation = hp.Choice('activation', values=['linear', 'relu', 'softmax'])

    optimizer = custom_optimizer(
        hp_learning_rate, decay_steps, hp_optimizer_choice)

    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape))

    for i in range(hp_num_cnn_layers):
        filters = hp.Choice(f'filters_{i}', values=[
            32, 64, 96, 128], default=hp_filters)
        kernel_size = hp.Choice(f'kernel_size_{i}', values=[
            1, 2, 3, 5], default=hp_kernel_size)

        model.add(Conv1D(filters=filters, kernel_size=kernel_size,
                         activation='relu', padding='same'))
        model.add(BatchNormalization())

        pool_size = ceil(model.layers[-1].output_shape[1] / 2)
        if pool_size > 1:
            model.add(MaxPooling1D(pool_size=pool_size))
        else:
            model.add(MaxPooling1D(pool_size=1))

        model.add(Dropout(rate=hp_dropout))

    for i in range(hp_num_bigru_layers):
        units = hp.Int(f'bigru_units_{i}', min_value=50,
                       max_value=200, step=50, default=hp_bigru_units)
        dropout = hp.Float(
            f'bigru_dropout_{i}', min_value=0.1, max_value=0.5, step=0.1, default=hp_dropout)
        return_sequences = i < hp_num_bigru_layers - 1

        model.add(
            Bidirectional(GRU(units=units, return_sequences=return_sequences, activation='relu')))
        model.add(Dropout(rate=dropout))

    model.add(Dense(units=forecast_period, activation=hp_activation, kernel_regularizer=l1_l2(
        hp_l1_regularizer, hp_l2_regularizer)))
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    model.summary()
    return model


'''-----------------------------Hybrid + Attention models-------------------------------'''


def build_cnn_lstm_attention_model(hp, input_shape, forecast_period):
    hp_num_cnn_layers = hp.Int(
        'num_cnn_layers', min_value=1, max_value=3, step=1)
    hp_num_lstm_layers = hp.Int(
        'num_lstm_layers', min_value=1, max_value=3, step=1)
    hp_filters = hp.Choice('filters', values=[32, 64, 96, 128])
    hp_kernel_size = hp.Choice('kernel_size', values=[1, 2, 3, 5])
    hp_lstm_units = hp.Int('lstm_units', min_value=50, max_value=200, step=50)
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    decay_steps = hp.Int('decay_steps', min_value=5000,
                         max_value=20000, step=5000)
    hp_optimizer_choice = hp.Choice(
        'optimizer', values=['adam', 'sgd', 'rmsprop'])
    hp_l1_regularizer = hp.Choice('l1_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_l2_regularizer = hp.Choice('l2_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_activation = hp.Choice('activation', values=['linear', 'relu', 'softmax'])

    optimizer = custom_optimizer(
        hp_learning_rate, decay_steps, hp_optimizer_choice)

    lookback = input_shape[0]

    inputs = Input(shape=input_shape)
    masked_inputs = Masking(mask_value=0.)(inputs)

    cnn_layers = []
    for i in range(hp_num_cnn_layers):
        filters = hp.Choice(f'filters_{i}', values=[
            32, 64, 96, 128], default=hp_filters)
        kernel_size = hp.Choice(f'kernel_size_{i}', values=[
            1, 2, 3, 5], default=hp_kernel_size)

        cnn_layers.append(Conv1D(
            filters=filters, kernel_size=kernel_size, activation='relu', padding='same'))
        cnn_layers.append(BatchNormalization())

    cnn_output = masked_inputs
    for layer in cnn_layers:
        cnn_output = layer(cnn_output)
        # We calculate pooling size based on output shape of the BatchNormalization layer
        pool_size = ceil(cnn_output.shape[1] / 2)
        if pool_size > 1:
            cnn_output = MaxPooling1D(pool_size=pool_size)(cnn_output)
        else:
            cnn_output = MaxPooling1D(pool_size=1)(cnn_output)

    lstm_layers = []
    for i in range(hp_num_lstm_layers):
        units = hp.Int(f'lstm_units_{i}', min_value=50,
                       max_value=200, step=50, default=hp_lstm_units)
        dropout = hp.Float(
            f'lstm_dropout_{i}', min_value=0.1, max_value=0.5, step=0.1, default=hp_dropout)

        lstm_layers.append(
            LSTM(units=units, return_sequences=True, activation='relu'))
        lstm_layers.append(Dropout(rate=dropout))

    lstm_output = cnn_output
    for layer in lstm_layers:
        lstm_output = layer(lstm_output)

    attention = dot([lstm_output, lstm_output], axes=[2, 2])
    attention = Dense(lookback, activation='relu')(attention)
    context = Concatenate(axis=-1)([cnn_output, lstm_output])
    flattened = Flatten()(context)
    output = Dense(forecast_period, activation=hp_activation, kernel_regularizer=l1_l2(
        hp_l1_regularizer, hp_l2_regularizer))(flattened)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    model.summary()
    return model


def build_cnn_gru_attention_model(hp, input_shape, forecast_period):
    hp_num_cnn_layers = hp.Int(
        'num_cnn_layers', min_value=1, max_value=3, step=1)
    hp_num_gru_layers = hp.Int(
        'num_gru_layers', min_value=1, max_value=3, step=1)
    hp_filters = hp.Choice('filters', values=[32, 64, 96, 128])
    hp_kernel_size = hp.Choice('kernel_size', values=[1, 2, 3, 5])
    hp_gru_units = hp.Int('gru_units', min_value=50, max_value=200, step=50)
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    decay_steps = hp.Int('decay_steps', min_value=5000,
                         max_value=20000, step=5000)
    hp_optimizer_choice = hp.Choice(
        'optimizer', values=['adam', 'sgd', 'rmsprop'])
    hp_l1_regularizer = hp.Choice('l1_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_l2_regularizer = hp.Choice('l2_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_activation = hp.Choice('activation', values=['linear', 'relu', 'softmax'])

    optimizer = custom_optimizer(
        hp_learning_rate, decay_steps, hp_optimizer_choice)

    lookback = input_shape[0]

    inputs = Input(shape=input_shape)
    masked_inputs = Masking(mask_value=0.)(inputs)

    cnn_layers = []
    for i in range(hp_num_cnn_layers):
        filters = hp.Choice(f'filters_{i}', values=[
            32, 64, 96, 128], default=hp_filters)
        kernel_size = hp.Choice(f'kernel_size_{i}', values=[
            1, 2, 3, 5], default=hp_kernel_size)

        cnn_layers.append(Conv1D(
            filters=filters, kernel_size=kernel_size, activation='relu', padding='same'))
        cnn_layers.append(BatchNormalization())

    cnn_output = masked_inputs
    for layer in cnn_layers:
        cnn_output = layer(cnn_output)
        # We calculate pooling size based on output shape of the BatchNormalization layer
        pool_size = ceil(cnn_output.shape[1] / 2)
        if pool_size > 1:
            cnn_output = MaxPooling1D(pool_size=pool_size)(cnn_output)
        else:
            cnn_output = MaxPooling1D(pool_size=1)(cnn_output)

    gru_layers = []
    for i in range(hp_num_gru_layers):
        units = hp.Int(f'gru_units_{i}', min_value=50,
                       max_value=200, step=50, default=hp_gru_units)
        dropout = hp.Float(
            f'gru_dropout_{i}', min_value=0.1, max_value=0.5, step=0.1, default=hp_dropout)

        gru_layers.append(
            GRU(units=units, return_sequences=True, activation='relu'))
        gru_layers.append(Dropout(rate=dropout))

    gru_output = cnn_output
    for layer in gru_layers:
        gru_output = layer(gru_output)

    attention = dot([gru_output, gru_output], axes=[2, 2])
    attention = Dense(lookback, activation='relu')(attention)
    context = Concatenate(axis=-1)([cnn_output, gru_output])
    flattened = Flatten()(context)
    output = Dense(forecast_period, activation=hp_activation, kernel_regularizer=l1_l2(
        hp_l1_regularizer, hp_l2_regularizer))(flattened)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    model.summary()
    return model


def build_cnn_bilstm_attention_model(hp, input_shape, forecast_period):
    hp_num_cnn_layers = hp.Int(
        'num_cnn_layers', min_value=1, max_value=3, step=1)
    hp_num_bilstm_layers = hp.Int(
        'num_bilstm_layers', min_value=1, max_value=3, step=1)
    hp_filters = hp.Choice('filters', values=[32, 64, 96, 128])
    hp_kernel_size = hp.Choice('kernel_size', values=[1, 2, 3, 5])
    hp_bilstm_units = hp.Int(
        'bilstm_units', min_value=50, max_value=200, step=50)
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    decay_steps = hp.Int('decay_steps', min_value=5000,
                         max_value=20000, step=5000)
    hp_optimizer_choice = hp.Choice(
        'optimizer', values=['adam', 'sgd', 'rmsprop'])
    hp_l1_regularizer = hp.Choice('l1_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_l2_regularizer = hp.Choice('l2_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_activation = hp.Choice('activation', values=['linear', 'relu', 'softmax'])

    optimizer = custom_optimizer(
        hp_learning_rate, decay_steps, hp_optimizer_choice)

    lookback = input_shape[0]

    inputs = Input(shape=input_shape)
    masked_inputs = Masking(mask_value=0.)(inputs)

    cnn_layers = []
    for i in range(hp_num_cnn_layers):
        filters = hp.Choice(f'filters_{i}', values=[
            32, 64, 96, 128], default=hp_filters)
        kernel_size = hp.Choice(f'kernel_size_{i}', values=[
            1, 2, 3, 5], default=hp_kernel_size)

        cnn_layers.append(Conv1D(
            filters=filters, kernel_size=kernel_size, activation='relu', padding='same'))
        cnn_layers.append(BatchNormalization())

    cnn_output = masked_inputs
    for layer in cnn_layers:
        cnn_output = layer(cnn_output)
        # We calculate pooling size based on output shape of the BatchNormalization layer
        pool_size = ceil(cnn_output.shape[1] / 2)
        if pool_size > 1:
            cnn_output = MaxPooling1D(pool_size=pool_size)(cnn_output)
        else:
            cnn_output = MaxPooling1D(pool_size=1)(cnn_output)

    bilstm_layers = []
    for i in range(hp_num_bilstm_layers):
        units = hp.Int(f'bilstm_units_{i}', min_value=50,
                       max_value=200, step=50, default=hp_bilstm_units)
        dropout = hp.Float(
            f'bilstm_dropout_{i}', min_value=0.1, max_value=0.5, step=0.1, default=hp_dropout)

        bilstm_layers.append(
            Bidirectional(LSTM(units=units, return_sequences=True, activation='relu')))
        bilstm_layers.append(Dropout(rate=dropout))

    bilstm_output = cnn_output
    for layer in bilstm_layers:
        bilstm_output = layer(bilstm_output)

    attention = dot([bilstm_output, bilstm_output], axes=[2, 2])
    attention = Dense(lookback, activation='relu')(attention)
    context = Concatenate(axis=-1)([cnn_output, bilstm_output])
    flattened = Flatten()(context)
    output = Dense(forecast_period, activation=hp_activation, kernel_regularizer=l1_l2(
        hp_l1_regularizer, hp_l2_regularizer))(flattened)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    model.summary()
    return model


def build_cnn_bigru_attention_model(hp, input_shape, forecast_period):
    hp_num_cnn_layers = hp.Int(
        'num_cnn_layers', min_value=1, max_value=3, step=1)
    hp_num_bigru_layers = hp.Int(
        'num_bigru_layers', min_value=1, max_value=3, step=1)
    hp_filters = hp.Choice('filters', values=[32, 64, 96, 128])
    hp_kernel_size = hp.Choice('kernel_size', values=[1, 2, 3, 5])
    hp_bigru_units = hp.Int('bigru_units', min_value=50,
                            max_value=200, step=50)
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    decay_steps = hp.Int('decay_steps', min_value=5000,
                         max_value=20000, step=5000)
    hp_optimizer_choice = hp.Choice(
        'optimizer', values=['adam', 'sgd', 'rmsprop'])
    hp_l1_regularizer = hp.Choice('l1_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_l2_regularizer = hp.Choice('l2_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_activation = hp.Choice('activation', values=['linear', 'relu', 'softmax'])

    optimizer = custom_optimizer(
        hp_learning_rate, decay_steps, hp_optimizer_choice)

    lookback = input_shape[0]

    inputs = Input(shape=input_shape)
    masked_inputs = Masking(mask_value=0.)(inputs)

    cnn_layers = []
    for i in range(hp_num_cnn_layers):
        filters = hp.Choice(f'filters_{i}', values=[
            32, 64, 96, 128], default=hp_filters)
        kernel_size = hp.Choice(f'kernel_size_{i}', values=[
            1, 2, 3, 5], default=hp_kernel_size)

        cnn_layers.append(Conv1D(
            filters=filters, kernel_size=kernel_size, activation='relu', padding='same'))
        cnn_layers.append(BatchNormalization())

    cnn_output = masked_inputs
    for layer in cnn_layers:
        cnn_output = layer(cnn_output)
        # We calculate pooling size based on output shape of the BatchNormalization layer
        pool_size = ceil(cnn_output.shape[1] / 2)
        if pool_size > 1:
            cnn_output = MaxPooling1D(pool_size=pool_size)(cnn_output)
        else:
            cnn_output = MaxPooling1D(pool_size=1)(cnn_output)

    bigru_layers = []
    for i in range(hp_num_bigru_layers):
        units = hp.Int(f'bigru_units_{i}', min_value=50,
                       max_value=200, step=50, default=hp_bigru_units)
        dropout = hp.Float(
            f'bigru_dropout_{i}', min_value=0.1, max_value=0.5, step=0.1, default=hp_dropout)

        bigru_layers.append(
            Bidirectional(GRU(units=units, return_sequences=True, activation='relu')))
        bigru_layers.append(Dropout(rate=dropout))

    bigru_output = cnn_output
    for layer in bigru_layers:
        bigru_output = layer(bigru_output)

    attention = dot([bigru_output, bigru_output], axes=[2, 2])
    attention = Dense(lookback, activation='relu')(attention)
    context = Concatenate(axis=-1)([cnn_output, bigru_output])
    flattened = Flatten()(context)
    output = Dense(forecast_period, activation=hp_activation, kernel_regularizer=l1_l2(
        hp_l1_regularizer, hp_l2_regularizer))(flattened)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    model.summary()
    return model


'''-----------------------------Deep Hybrid + Attention models-------------------------------'''


def build_cnn_attention_lstm_model(hp, input_shape, forecast_period):
    hp_num_cnn_layers = hp.Int(
        'num_cnn_layers', min_value=1, max_value=3, step=1)
    hp_num_lstm_layers = hp.Int(
        'num_lstm_layers', min_value=1, max_value=3, step=1)
    hp_filters = hp.Choice('filters', values=[32, 64, 96, 128])
    hp_kernel_size = hp.Choice('kernel_size', values=[1, 2, 3, 5])
    hp_lstm_units = hp.Int('lstm_units', min_value=50, max_value=200, step=50)
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    decay_steps = hp.Int('decay_steps', min_value=5000,
                         max_value=20000, step=5000)
    hp_optimizer_choice = hp.Choice(
        'optimizer', values=['adam', 'sgd', 'rmsprop'])
    hp_l1_regularizer = hp.Choice('l1_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_l2_regularizer = hp.Choice('l2_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_activation = hp.Choice('activation', values=['linear', 'relu', 'softmax'])

    optimizer = custom_optimizer(
        hp_learning_rate, decay_steps, hp_optimizer_choice)

    inputs = Input(shape=input_shape)
    masked_inputs = Masking(mask_value=0.)(inputs)

    cnn_layers = []
    for i in range(hp_num_cnn_layers):
        filters = hp.Choice(f'filters_{i}', values=[
            32, 64, 96, 128], default=hp_filters)
        kernel_size = hp.Choice(f'kernel_size_{i}', values=[
            1, 2, 3, 5], default=hp_kernel_size)

        cnn_layers.append(Conv1D(
            filters=filters, kernel_size=kernel_size, activation='relu', padding='same'))
        cnn_layers.append(BatchNormalization())

    cnn_output = masked_inputs
    for layer in cnn_layers:
        cnn_output = layer(cnn_output)
        # We calculate pooling size based on output shape of the BatchNormalization layer
        pool_size = ceil(cnn_output.shape[1] / 2)
        if pool_size > 1:
            cnn_output = MaxPooling1D(pool_size=pool_size)(cnn_output)
        else:
            cnn_output = MaxPooling1D(pool_size=1)(cnn_output)

    attention = Attention()([cnn_output, cnn_output])

    lstm_layers = []
    for i in range(hp_num_lstm_layers):
        units = hp.Int(f'lstm_units_{i}', min_value=50,
                       max_value=200, step=50, default=hp_lstm_units)
        dropout = hp.Float(
            f'lstm_dropout_{i}', min_value=0.1, max_value=0.5, step=0.1, default=hp_dropout)

        lstm_layers.append(
            LSTM(units=units, return_sequences=True, activation='relu'))
        lstm_layers.append(Dropout(rate=dropout))

    lstm_output = attention
    for layer in lstm_layers:
        lstm_output = layer(lstm_output)

    output = Dense(forecast_period, activation=hp_activation, kernel_regularizer=l1_l2(
        hp_l1_regularizer, hp_l2_regularizer))(lstm_output)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    model.summary()
    return model


def build_cnn_attention_gru_model(hp, input_shape, forecast_period):
    hp_num_cnn_layers = hp.Int(
        'num_cnn_layers', min_value=1, max_value=3, step=1)
    hp_num_gru_layers = hp.Int(
        'num_gru_layers', min_value=1, max_value=3, step=1)
    hp_filters = hp.Choice('filters', values=[32, 64, 96, 128])
    hp_kernel_size = hp.Choice('kernel_size', values=[1, 2, 3, 5])
    hp_gru_units = hp.Int('gru_units', min_value=50, max_value=200, step=50)
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    decay_steps = hp.Int('decay_steps', min_value=5000,
                         max_value=20000, step=5000)
    hp_optimizer_choice = hp.Choice(
        'optimizer', values=['adam', 'sgd', 'rmsprop'])
    hp_l1_regularizer = hp.Choice('l1_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_l2_regularizer = hp.Choice('l2_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_activation = hp.Choice('activation', values=['linear', 'relu', 'softmax'])

    optimizer = custom_optimizer(
        hp_learning_rate, decay_steps, hp_optimizer_choice)

    inputs = Input(shape=input_shape)
    masked_inputs = Masking(mask_value=0.)(inputs)

    cnn_layers = []
    for i in range(hp_num_cnn_layers):
        filters = hp.Choice(f'filters_{i}', values=[
            32, 64, 96, 128], default=hp_filters)
        kernel_size = hp.Choice(f'kernel_size_{i}', values=[
            1, 2, 3, 5], default=hp_kernel_size)

        cnn_layers.append(Conv1D(
            filters=filters, kernel_size=kernel_size, activation='relu', padding='same'))
        cnn_layers.append(BatchNormalization())

    cnn_output = masked_inputs
    for layer in cnn_layers:
        cnn_output = layer(cnn_output)
        # We calculate pooling size based on output shape of the BatchNormalization layer
        pool_size = ceil(cnn_output.shape[1] / 2)
        if pool_size > 1:
            cnn_output = MaxPooling1D(pool_size=pool_size)(cnn_output)
        else:
            cnn_output = MaxPooling1D(pool_size=1)(cnn_output)

    attention = Attention()([cnn_output, cnn_output])

    gru_layers = []
    for i in range(hp_num_gru_layers):
        units = hp.Int(f'gru_units_{i}', min_value=50,
                       max_value=200, step=50, default=hp_gru_units)
        dropout = hp.Float(
            f'gru_dropout_{i}', min_value=0.1, max_value=0.5, step=0.1, default=hp_dropout)

        gru_layers.append(
            GRU(units=units, return_sequences=True, activation='relu'))
        gru_layers.append(Dropout(rate=dropout))

    gru_output = attention
    for layer in gru_layers:
        gru_output = layer(gru_output)

    output = Dense(forecast_period, activation=hp_activation, kernel_regularizer=l1_l2(
        hp_l1_regularizer, hp_l2_regularizer))(gru_output)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    model.summary()
    return model


def build_cnn_attention_bilstm_model(hp, input_shape, forecast_period):
    hp_num_cnn_layers = hp.Int(
        'num_cnn_layers', min_value=1, max_value=3, step=1)
    hp_num_bilstm_layers = hp.Int(
        'num_bilstm_layers', min_value=1, max_value=3, step=1)
    hp_filters = hp.Choice('filters', values=[32, 64, 96, 128])
    hp_kernel_size = hp.Choice('kernel_size', values=[1, 2, 3, 5])
    hp_bilstm_units = hp.Int(
        'bilstm_units', min_value=50, max_value=200, step=50)
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    decay_steps = hp.Int('decay_steps', min_value=5000,
                         max_value=20000, step=5000)
    hp_optimizer_choice = hp.Choice(
        'optimizer', values=['adam', 'sgd', 'rmsprop'])
    hp_l1_regularizer = hp.Choice('l1_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_l2_regularizer = hp.Choice('l2_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_activation = hp.Choice('activation', values=['linear', 'relu', 'softmax'])

    optimizer = custom_optimizer(
        hp_learning_rate, decay_steps, hp_optimizer_choice)

    inputs = Input(shape=input_shape)
    masked_inputs = Masking(mask_value=0.)(inputs)

    cnn_layers = []
    for i in range(hp_num_cnn_layers):
        filters = hp.Choice(f'filters_{i}', values=[
            32, 64, 96, 128], default=hp_filters)
        kernel_size = hp.Choice(f'kernel_size_{i}', values=[
            1, 2, 3, 5], default=hp_kernel_size)

        cnn_layers.append(Conv1D(
            filters=filters, kernel_size=kernel_size, activation='relu', padding='same'))
        cnn_layers.append(BatchNormalization())

    cnn_output = masked_inputs
    for layer in cnn_layers:
        cnn_output = layer(cnn_output)
        # We calculate pooling size based on output shape of the BatchNormalization layer
        pool_size = ceil(cnn_output.shape[1] / 2)
        if pool_size > 1:
            cnn_output = MaxPooling1D(pool_size=pool_size)(cnn_output)
        else:
            cnn_output = MaxPooling1D(pool_size=1)(cnn_output)

    attention = Attention()([cnn_output, cnn_output])

    bilstm_layers = []
    for i in range(hp_num_bilstm_layers):
        units = hp.Int(f'bilstm_units_{i}', min_value=50,
                       max_value=200, step=50, default=hp_bilstm_units)
        dropout = hp.Float(
            f'bilstm_dropout_{i}', min_value=0.1, max_value=0.5, step=0.1, default=hp_dropout)

        bilstm_layers.append(
            Bidirectional(LSTM(units=units, return_sequences=True, activation='relu')))
        bilstm_layers.append(Dropout(rate=dropout))

    bilstm_output = attention
    for layer in bilstm_layers:
        bilstm_output = layer(bilstm_output)

    output = Dense(forecast_period, activation=hp_activation, kernel_regularizer=l1_l2(
        hp_l1_regularizer, hp_l2_regularizer))(bilstm_output)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    model.summary()
    return model


def build_cnn_attention_bigru_model(hp, input_shape, forecast_period):
    hp_num_cnn_layers = hp.Int(
        'num_cnn_layers', min_value=1, max_value=3, step=1)
    hp_num_bigru_layers = hp.Int(
        'num_bigru_layers', min_value=1, max_value=3, step=1)
    hp_filters = hp.Choice('filters', values=[32, 64, 96, 128])
    hp_kernel_size = hp.Choice('kernel_size', values=[1, 2, 3, 5])
    hp_bigru_units = hp.Int('bigru_units', min_value=50,
                            max_value=200, step=50)
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    decay_steps = hp.Int('decay_steps', min_value=5000,
                         max_value=20000, step=5000)
    hp_optimizer_choice = hp.Choice(
        'optimizer', values=['adam', 'sgd', 'rmsprop'])
    hp_l1_regularizer = hp.Choice('l1_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_l2_regularizer = hp.Choice('l2_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_activation = hp.Choice('activation', values=['linear', 'relu', 'softmax'])

    optimizer = custom_optimizer(
        hp_learning_rate, decay_steps, hp_optimizer_choice)

    inputs = Input(shape=input_shape)
    masked_inputs = Masking(mask_value=0.)(inputs)

    cnn_layers = []
    for i in range(hp_num_cnn_layers):
        filters = hp.Choice(f'filters_{i}', values=[
            32, 64, 96, 128], default=hp_filters)
        kernel_size = hp.Choice(f'kernel_size_{i}', values=[
            1, 2, 3, 5], default=hp_kernel_size)

        cnn_layers.append(Conv1D(
            filters=filters, kernel_size=kernel_size, activation='relu', padding='same'))
        cnn_layers.append(BatchNormalization())

    cnn_output = masked_inputs
    for layer in cnn_layers:
        cnn_output = layer(cnn_output)
        # We calculate pooling size based on output shape of the BatchNormalization layer
        pool_size = ceil(cnn_output.shape[1] / 2)
        if pool_size > 1:
            cnn_output = MaxPooling1D(pool_size=pool_size)(cnn_output)
        else:
            cnn_output = MaxPooling1D(pool_size=1)(cnn_output)

    attention = Attention()([cnn_output, cnn_output])

    bigru_layers = []
    for i in range(hp_num_bigru_layers):
        units = hp.Int(f'bigru_units_{i}', min_value=50,
                       max_value=200, step=50, default=hp_bigru_units)
        dropout = hp.Float(
            f'bigru_dropout_{i}', min_value=0.1, max_value=0.5, step=0.1, default=hp_dropout)

        bigru_layers.append(
            Bidirectional(GRU(units=units, return_sequences=True, activation='relu')))
        bigru_layers.append(Dropout(rate=dropout))

    bigru_output = attention
    for layer in bigru_layers:
        bigru_output = layer(bigru_output)

    output = Dense(forecast_period, activation=hp_activation, kernel_regularizer=l1_l2(
        hp_l1_regularizer, hp_l2_regularizer))(bigru_output)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    model.summary()
    return model


'''-----------------------------Deep More Hybrid + Attention models-------------------------------'''


def build_cnn_attention_lstm_attention_model(hp, input_shape, forecast_period):
    hp_num_cnn_layers = hp.Int(
        'num_cnn_layers', min_value=1, max_value=3, step=1)
    hp_num_lstm_layers = hp.Int(
        'num_lstm_layers', min_value=1, max_value=3, step=1)
    hp_filters = hp.Choice('filters', values=[32, 64, 96, 128])
    hp_kernel_size = hp.Choice('kernel_size', values=[1, 2, 3, 5])
    hp_lstm_units = hp.Int('lstm_units', min_value=50, max_value=200, step=50)
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    decay_steps = hp.Int('decay_steps', min_value=5000,
                         max_value=20000, step=5000)
    hp_optimizer_choice = hp.Choice(
        'optimizer', values=['adam', 'sgd', 'rmsprop'])
    hp_l1_regularizer = hp.Choice('l1_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_l2_regularizer = hp.Choice('l2_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_activation = hp.Choice('activation', values=['linear', 'relu', 'softmax'])

    optimizer = custom_optimizer(
        hp_learning_rate, decay_steps, hp_optimizer_choice)

    inputs = Input(shape=input_shape)
    masked_inputs = Masking(mask_value=0.)(inputs)

    cnn_layers = []
    for i in range(hp_num_cnn_layers):
        filters = hp.Choice(f'filters_{i}', values=[
            32, 64, 96, 128], default=hp_filters)
        kernel_size = hp.Choice(f'kernel_size_{i}', values=[
            1, 2, 3, 5], default=hp_kernel_size)

        cnn_layers.append(Conv1D(
            filters=filters, kernel_size=kernel_size, activation='relu', padding='same'))
        cnn_layers.append(BatchNormalization())

    cnn_output = masked_inputs
    for layer in cnn_layers:
        cnn_output = layer(cnn_output)
        # We calculate pooling size based on output shape of the BatchNormalization layer
        pool_size = ceil(cnn_output.shape[1] / 2)
        if pool_size > 1:
            cnn_output = MaxPooling1D(pool_size=pool_size)(cnn_output)
        else:
            cnn_output = MaxPooling1D(pool_size=1)(cnn_output)

    attention1 = Attention()([cnn_output, cnn_output])

    lstm_layers = []
    for i in range(hp_num_lstm_layers):
        units = hp.Int(f'lstm_units_{i}', min_value=50,
                       max_value=200, step=50, default=hp_lstm_units)
        dropout = hp.Float(
            f'lstm_dropout_{i}', min_value=0.1, max_value=0.5, step=0.1, default=hp_dropout)

        lstm_layers.append(
            LSTM(units=units, return_sequences=True, activation='relu'))
        lstm_layers.append(Dropout(rate=dropout))

    lstm_output = attention1
    for layer in lstm_layers:
        lstm_output = layer(lstm_output)

    attention2 = Attention()([lstm_output, lstm_output])

    output = Dense(forecast_period, activation=hp_activation, kernel_regularizer=l1_l2(
        hp_l1_regularizer, hp_l2_regularizer))(attention2)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    model.summary()
    return model


def build_cnn_attention_gru_attention_model(hp, input_shape, forecast_period):
    hp_num_cnn_layers = hp.Int(
        'num_cnn_layers', min_value=1, max_value=3, step=1)
    hp_num_gru_layers = hp.Int(
        'num_gru_layers', min_value=1, max_value=3, step=1)
    hp_filters = hp.Choice('filters', values=[32, 64, 96, 128])
    hp_kernel_size = hp.Choice('kernel_size', values=[1, 2, 3, 5])
    hp_gru_units = hp.Int('gru_units', min_value=50, max_value=200, step=50)
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    decay_steps = hp.Int('decay_steps', min_value=5000,
                         max_value=20000, step=5000)
    hp_optimizer_choice = hp.Choice(
        'optimizer', values=['adam', 'sgd', 'rmsprop'])
    hp_l1_regularizer = hp.Choice('l1_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_l2_regularizer = hp.Choice('l2_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_activation = hp.Choice('activation', values=['linear', 'relu', 'softmax'])

    optimizer = custom_optimizer(
        hp_learning_rate, decay_steps, hp_optimizer_choice)

    inputs = Input(shape=input_shape)
    masked_inputs = Masking(mask_value=0.)(inputs)

    cnn_layers = []
    for i in range(hp_num_cnn_layers):
        filters = hp.Choice(f'filters_{i}', values=[
            32, 64, 96, 128], default=hp_filters)
        kernel_size = hp.Choice(f'kernel_size_{i}', values=[
            1, 2, 3, 5], default=hp_kernel_size)

        cnn_layers.append(Conv1D(
            filters=filters, kernel_size=kernel_size, activation='relu', padding='same'))
        cnn_layers.append(BatchNormalization())

    cnn_output = masked_inputs
    for layer in cnn_layers:
        cnn_output = layer(cnn_output)
        # We calculate pooling size based on output shape of the BatchNormalization layer
        pool_size = ceil(cnn_output.shape[1] / 2)
        if pool_size > 1:
            cnn_output = MaxPooling1D(pool_size=pool_size)(cnn_output)
        else:
            cnn_output = MaxPooling1D(pool_size=1)(cnn_output)

    attention1 = Attention()([cnn_output, cnn_output])

    gru_layers = []
    for i in range(hp_num_gru_layers):
        units = hp.Int(f'gru_units_{i}', min_value=50,
                       max_value=200, step=50, default=hp_gru_units)
        dropout = hp.Float(
            f'gru_dropout_{i}', min_value=0.1, max_value=0.5, step=0.1, default=hp_dropout)

        gru_layers.append(
            GRU(units=units, return_sequences=True, activation='relu'))
        gru_layers.append(Dropout(rate=dropout))

    gru_output = attention1
    for layer in gru_layers:
        gru_output = layer(gru_output)

    attention2 = Attention()([gru_output, gru_output])

    output = Dense(forecast_period, activation=hp_activation, kernel_regularizer=l1_l2(
        hp_l1_regularizer, hp_l2_regularizer))(attention2)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    model.summary()
    return model


def build_cnn_attention_bilstm_attention_model(hp, input_shape, forecast_period):
    hp_num_cnn_layers = hp.Int(
        'num_cnn_layers', min_value=1, max_value=3, step=1)
    hp_num_bilstm_layers = hp.Int(
        'num_bilstm_layers', min_value=1, max_value=3, step=1)
    hp_filters = hp.Choice('filters', values=[32, 64, 96, 128])
    hp_kernel_size = hp.Choice('kernel_size', values=[1, 2, 3, 5])
    hp_bilstm_units = hp.Int('bilstm_units', min_value=50, max_value=200, step=50)
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    decay_steps = hp.Int('decay_steps', min_value=5000,
                         max_value=20000, step=5000)
    hp_optimizer_choice = hp.Choice(
        'optimizer', values=['adam', 'sgd', 'rmsprop'])
    hp_l1_regularizer = hp.Choice('l1_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_l2_regularizer = hp.Choice('l2_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_activation = hp.Choice('activation', values=['linear', 'relu', 'softmax'])

    optimizer = custom_optimizer(
        hp_learning_rate, decay_steps, hp_optimizer_choice)

    lookback = input_shape[0]

    inputs = Input(shape=input_shape)
    masked_inputs = Masking(mask_value=0.)(inputs)

    cnn_layers = []
    for i in range(hp_num_cnn_layers):
        filters = hp.Choice(f'filters_{i}', values=[
            32, 64, 96, 128], default=hp_filters)
        kernel_size = hp.Choice(f'kernel_size_{i}', values=[
            1, 2, 3, 5], default=hp_kernel_size)

        cnn_layers.append(Conv1D(
            filters=filters, kernel_size=kernel_size, activation='relu', padding='same'))
        cnn_layers.append(BatchNormalization())

    cnn_output = masked_inputs
    for layer in cnn_layers:
        cnn_output = layer(cnn_output)
        # We calculate pooling size based on output shape of the BatchNormalization layer
        pool_size = ceil(cnn_output.shape[1] / 2)
        if pool_size > 1:
            cnn_output = MaxPooling1D(pool_size=pool_size)(cnn_output)
        else:
            cnn_output = MaxPooling1D(pool_size=1)(cnn_output)

    attention1 = Attention()([cnn_output, cnn_output])

    bilstm_layers = []
    for i in range(hp_num_bilstm_layers):
        units = hp.Int(f'bilstm_units_{i}', min_value=50,
                       max_value=200, step=50, default=hp_bilstm_units)
        dropout = hp.Float(
            f'bilstm_dropout_{i}', min_value=0.1, max_value=0.5, step=0.1, default=hp_dropout)

        bilstm_layers.append(
            Bidirectional(LSTM(units=units, return_sequences=True, activation='relu')))
        bilstm_layers.append(Dropout(rate=dropout))

    bilstm_output = attention1
    for layer in bilstm_layers:
        bilstm_output = layer(bilstm_output)

    attention2 = Attention()([bilstm_output, bilstm_output])

    output = Dense(forecast_period, activation=hp_activation, kernel_regularizer=l1_l2(
        hp_l1_regularizer, hp_l2_regularizer))(attention2)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    model.summary()
    return model


def build_cnn_attention_bigru_attention_model(hp, input_shape, forecast_period):
    hp_num_cnn_layers = hp.Int(
        'num_cnn_layers', min_value=1, max_value=3, step=1)
    hp_num_bigru_layers = hp.Int(
        'num_bigru_layers', min_value=1, max_value=3, step=1)
    hp_filters = hp.Choice('filters', values=[32, 64, 96, 128])
    hp_kernel_size = hp.Choice('kernel_size', values=[1, 2, 3, 5])
    hp_bigru_units = hp.Int('bigru_units', min_value=50, max_value=200, step=50)
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    decay_steps = hp.Int('decay_steps', min_value=5000,
                         max_value=20000, step=5000)
    hp_optimizer_choice = hp.Choice(
        'optimizer', values=['adam', 'sgd', 'rmsprop'])
    hp_l1_regularizer = hp.Choice('l1_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_l2_regularizer = hp.Choice('l2_regularizer', values=[0.00001, 0.001, 0.01, 0.1])
    hp_activation = hp.Choice('activation', values=['linear', 'relu', 'softmax'])

    optimizer = custom_optimizer(
        hp_learning_rate, decay_steps, hp_optimizer_choice)

    lookback = input_shape[0]

    inputs = Input(shape=input_shape)
    masked_inputs = Masking(mask_value=0.)(inputs)

    cnn_layers = []
    for i in range(hp_num_cnn_layers):
        filters = hp.Choice(f'filters_{i}', values=[
            32, 64, 96, 128], default=hp_filters)
        kernel_size = hp.Choice(f'kernel_size_{i}', values=[
            1, 2, 3, 5], default=hp_kernel_size)

        cnn_layers.append(Conv1D(
            filters=filters, kernel_size=kernel_size, activation='relu', padding='same'))
        cnn_layers.append(BatchNormalization())

    cnn_output = masked_inputs
    for layer in cnn_layers:
        cnn_output = layer(cnn_output)
        # We calculate pooling size based on output shape of the BatchNormalization layer
        pool_size = ceil(cnn_output.shape[1] / 2)
        if pool_size > 1:
            cnn_output = MaxPooling1D(pool_size=pool_size)(cnn_output)
        else:
            cnn_output = MaxPooling1D(pool_size=1)(cnn_output)

    attention1 = Attention()([cnn_output, cnn_output])

    bigru_layers = []
    for i in range(hp_num_bigru_layers):
        units = hp.Int(f'bigru_units_{i}', min_value=50,
                       max_value=200, step=50, default=hp_bigru_units)
        dropout = hp.Float(
            f'bigru_dropout_{i}', min_value=0.1, max_value=0.5, step=0.1, default=hp_dropout)

        bigru_layers.append(
            Bidirectional(GRU(units=units, return_sequences=True, activation='relu')))
        bigru_layers.append(Dropout(rate=dropout))

    bigru_output = attention1
    for layer in bigru_layers:
        bigru_output = layer(bigru_output)

    attention2 = Attention()([bigru_output, bigru_output])

    output = Dense(forecast_period, activation=hp_activation, kernel_regularizer=l1_l2(
        hp_l1_regularizer, hp_l2_regularizer))(attention2)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    model.summary()
    return model


def build_model(hp, model_type, input_shape, forecast_period):
    # -----------------------------Simple models-------------------------------
    if model_type == LSTM_MODEL:
        return build_lstm_model(hp, input_shape, forecast_period)
    elif model_type == GRU_MODEL:
        return build_gru_model(hp, input_shape, forecast_period)
    elif model_type == CNN_MODEL:
        return build_cnn_model(hp, input_shape, forecast_period)

    # -----------------------------Simple Bi models-------------------------------
    elif model_type == BiLSTM_MODEL:
        return build_bilstm_model(hp, input_shape, forecast_period)
    elif model_type == BiGRU_MODEL:
        return build_bigru_model(hp, input_shape, forecast_period)

    # -----------------------------Simple + Attention models-------------------------------
    elif model_type == LSTM_ATTENTION_MODEL:
        return build_lstm_attention_model(hp, input_shape, forecast_period)
    elif model_type == GRU_ATTENTION_MODEL:
        return build_gru_attention_model(hp, input_shape, forecast_period)
    elif model_type == CNN_ATTENTION_MODEL:
        return build_cnn_attention_model(hp, input_shape, forecast_period)
    # -----------------------------Bi + Attention models-------------------------------
    elif model_type == BiLSTM_ATTENTION_MODEL:
        return build_bilstm_attention_model(hp, input_shape, forecast_period)
    elif model_type == BiGRU_ATTENTION_MODEL:
        return build_bigru_attention_model(hp, input_shape, forecast_period)
    # -----------------------------Hybrid models-------------------------------
    elif model_type == CNN_LSTM_MODEL:
        return build_cnn_lstm_model(hp, input_shape, forecast_period)
    elif model_type == CNN_GRU_MODEL:
        return build_cnn_gru_model(hp, input_shape, forecast_period)
    elif model_type == CNN_BiLSTM_MODEL:
        return build_cnn_bilstm_model(hp, input_shape, forecast_period)
    elif model_type == CNN_BiGRU_MODEL:
        return build_cnn_bigru_model(hp, input_shape, forecast_period)
    # -----------------------------Hybrid + Attention models-------------------------------
    elif model_type == CNN_LSTM_ATTENTION_MODEL:
        return build_cnn_lstm_attention_model(hp, input_shape, forecast_period)
    elif model_type == CNN_GRU_ATTENTION_MODEL:
        return build_cnn_gru_attention_model(hp, input_shape, forecast_period)
    elif model_type == CNN_BiLSTM_ATTENTION_MODEL:
        return build_cnn_bilstm_attention_model(hp, input_shape, forecast_period)
    elif model_type == CNN_BiGRU_ATTENTION_MODEL:
        return build_cnn_bigru_attention_model(hp, input_shape, forecast_period)
    # -----------------------------Deep Hybrid + Attention models-------------------------------
    elif model_type == CNN_ATTENTION_LSTM_MODEL:
        return build_cnn_attention_lstm_model(hp, input_shape, forecast_period)
    elif model_type == CNN_ATTENTION_GRU_MODEL:
        return build_cnn_attention_gru_model(hp, input_shape, forecast_period)
    elif model_type == CNN_ATTENTION_BiLSTM_MODEL:
        return build_cnn_attention_bilstm_model(hp, input_shape, forecast_period)
    elif model_type == CNN_ATTENTION_BiGRU_MODEL:
        return build_cnn_attention_bigru_model(hp, input_shape, forecast_period)
    # -----------------------------Deep More Hybrid + Attention models-------------------------------
    elif model_type == CNN_ATTENTION_LSTM_ATTENTION_MODEL:
        return build_cnn_attention_lstm_attention_model(hp, input_shape, forecast_period)
    elif model_type == CNN_ATTENTION_GRU_ATTENTION_MODEL:
        return build_cnn_attention_gru_attention_model(hp, input_shape, forecast_period)
    elif model_type == CNN_ATTENTION_BiLSTM_ATTENTION_MODEL:
        return build_cnn_attention_bilstm_attention_model(hp, input_shape, forecast_period)
    elif model_type == CNN_ATTENTION_BiGRU_ATTENTION_MODEL:
        return build_cnn_attention_bigru_attention_model(hp, input_shape, forecast_period)
    else:
        raise ValueError(
            "Invalid model type. Please choose from the available models.")


def tune_custom_model(dataX, input_shape, forecast_period, model_type, band):
    x_train, y_train, x_val, y_val = dataX

    def build_hypermodel(hp):
        return build_model(hp, model_type, input_shape, forecast_period)

    tuner = Hyperband(
        build_hypermodel,
        objective='val_loss',
        max_epochs=EPOCH,
        factor=3,
        hyperband_iterations=3,
        directory=band,
        project_name=model_type,
    )

    # tuner.search_space_summary()
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    tuner.search(x_train, y_train, epochs=EPOCH, validation_data=(
        x_val, y_val), callbacks=[early_stopping])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    return best_hps
