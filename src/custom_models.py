#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 17:49:19 2024

@author: mesabo
"""
from math import ceil
import  numpy as np
from sklearn.model_selection import GridSearchCV

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Flatten, Conv1D, MaxPooling1D, GRU,
                          Bidirectional, Input,
                          Masking, Concatenate, dot,
                          Permute, Dropout, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
# from scikeras.wrappers import KerasRegressor

from tensorflow.keras.regularizers import l1, l2

from tensorflow.keras.optimizers.schedules import ExponentialDecay

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
    EPOCH, BATCH_SIZE, CHECK_PATH, PARAMS_GRID
)


def train_model(model, trainX, trainY, valX, valY):
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ]
    print(f"[---------------TRAINING MODEL---------------]\n")
    history = model.fit(trainX, trainY, epochs=EPOCH, batch_size=BATCH_SIZE, validation_data=(valX, valY),
                        callbacks=callbacks)
    return history


def make_predictions(model, testX, testY, scaler):
    testPredict = model.predict(testX)

    testPredict = scaler.inverse_transform(testPredict)
    testOutput = scaler.inverse_transform(testY)

    return testPredict, testOutput


def custom_optimizer(train_size, optimizer_type='adam'):
    steps_per_epoch = train_size // BATCH_SIZE
    initial_learning_rate = 1e-2
    decay_steps = 10000
    decay_rate = 0.9

    lr_schedule = ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_rate * steps_per_epoch,
        decay_rate=decay_rate
    )

    if optimizer_type == 'adam':
        optimizer = Adam(learning_rate=lr_schedule, amsgrad=True)
    elif optimizer_type == 'sgd':
        optimizer = SGD(learning_rate=lr_schedule, momentum=0.9)
    elif optimizer_type == 'rmsprop':
        optimizer = RMSprop(learning_rate=lr_schedule)
    else:
        raise ValueError("Unknown optimizer type. Supported types are 'adam', 'sgd', and 'rmsprop'.")

    return optimizer


def build_model(train_size, model_type, input_shape, forecast_period):
    # -----------------------------Simple models-------------------------------
    if model_type == LSTM_MODEL:
        return build_lstm_model(train_size, input_shape, forecast_period)
    elif model_type == GRU_MODEL:
        return build_gru_model(train_size, input_shape, forecast_period)
    elif model_type == CNN_MODEL:
        # return build_cnn_model(train_size, input_shape, forecast_period)
        return build_final_cnn_model(input_shape, forecast_period)
    # -----------------------------Simple Bi models-------------------------------
    elif model_type == BiLSTM_MODEL:
        return build_bilstm_model(train_size, input_shape, forecast_period)
    elif model_type == BiGRU_MODEL:
        return build_bigru_model(train_size, input_shape, forecast_period)
    # -----------------------------Simple + Attention models-------------------------------
    elif model_type == LSTM_ATTENTION_MODEL:
        return build_lstm_attention_model(train_size, input_shape, forecast_period)
    elif model_type == GRU_ATTENTION_MODEL:
        return build_gru_attention_model(train_size, input_shape, forecast_period)
    elif model_type == CNN_ATTENTION_MODEL:
        return build_cnn_attention_model(train_size, input_shape, forecast_period)
    # -----------------------------Bi + Attention models-------------------------------
    elif model_type == BiLSTM_ATTENTION_MODEL:
        return build_bilstm_attention_model(train_size, input_shape, forecast_period)
    elif model_type == BiGRU_ATTENTION_MODEL:
        return build_bigru_attention_model(train_size, input_shape, forecast_period)
    # -----------------------------Hybrid models-------------------------------
    elif model_type == CNN_LSTM_MODEL:
        return build_cnn_lstm_model(train_size, input_shape, forecast_period)
    elif model_type == CNN_GRU_MODEL:
        return build_cnn_gru_model(train_size, input_shape, forecast_period)
    elif model_type == CNN_BiLSTM_MODEL:
        return build_cnn_bilstm_model(train_size, input_shape, forecast_period)
    elif model_type == CNN_BiGRU_MODEL:
        return build_cnn_bigru_model(train_size, input_shape, forecast_period)
    # -----------------------------Hybrid + Attention models-------------------------------
    elif model_type == CNN_LSTM_ATTENTION_MODEL:
        return build_cnn_lstm_attention_model(train_size, input_shape, forecast_period)
    elif model_type == CNN_GRU_ATTENTION_MODEL:
        return build_cnn_gru_attention_model(train_size, input_shape, forecast_period)
    elif model_type == CNN_BiLSTM_ATTENTION_MODEL:
        return build_cnn_bilstm_attention_model(train_size, input_shape, forecast_period)
    elif model_type == CNN_BiGRU_ATTENTION_MODEL:
        return build_cnn_bigru_attention_model(train_size, input_shape, forecast_period)
    # -----------------------------Deep Hybrid + Attention models-------------------------------
    elif model_type == CNN_ATTENTION_LSTM_MODEL:
        return build_cnn_attention_lstm_model(train_size, input_shape, forecast_period)
    elif model_type == CNN_ATTENTION_GRU_MODEL:
        return build_cnn_attention_gru_model(train_size, input_shape, forecast_period)
    elif model_type == CNN_ATTENTION_BiLSTM_MODEL:
        return build_cnn_attention_bilstm_model(train_size, input_shape, forecast_period)
    elif model_type == CNN_ATTENTION_BiGRU_MODEL:
        return build_cnn_attention_bigru_model(train_size, input_shape, forecast_period)
    # -----------------------------Deep More Hybrid + Attention models-------------------------------
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


# mean_absolute_error or mean_squared_error


'''-----------------------------Simple models-------------------------------'''


def build_lstm_model(train_size, input_shape, forecast_period, optimizer='adam'):
    optimizer = custom_optimizer(train_size=train_size)

    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape))
    model.add(LSTM(200, return_sequences=True, activation='relu'))
    model.add(Dropout(0.3))
    model.add(LSTM(200, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(forecast_period, activation='softmax'))
    model.compile(loss='mean_absolute_error', optimizer=optimizer)
    return model


def tune_hyperparameters(input_shape, model_type, forecast_period, trainX, trainY):
    hyper_model = build_model(len(trainX), model_type, input_shape, forecast_period)

    model = KerasRegressor(build_fn=lambda: hyper_model, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=0)

    ts, mt, iz, fp = [len(trainX)], [model_type], [input_shape], [forecast_period]
    batch_size = [10, 20, 40, 60, 80, 100]
    epochs = [10, 50, 100]
    optimizer = ['adam', 'sgd', 'rmsprop'],
    param_grid = dict(batch_size=batch_size, epochs=epochs,
                      # train_size=ts,
                      # model_type=mt, input_shape=iz, forecast_period=fp,
                      optimizer=optimizer)

    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error', verbose=2)
    grid_result = grid.fit(trainX, trainY)

    # Summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    return grid_result.best_params_


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
    model.add(Dense(32, activation='softmax'))
    model.add(Dense(forecast_period))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


from keras.regularizers import l1, l2

# Best parameters obtained from hyperparameter tuning
best_params = {
    "num_layers": 2,
    "learning_rate": 0.001,
    "decay_steps": 20000,
    "optimizer": "rmsprop",
    "dropout_rate": 0.1,
    "l1_regularizer": 0.001,
    "l2_regularizer": 0.001,
    "filters_0": 128,
    "kernel_size_0": 3,
    "filters_1": 128,
    "kernel_size_1": 1,
    "filters_2": 32,
    "kernel_size_2": 1,
}


def c_optimizer(hp_learning_rate, decay_steps, hp_optimizer_choice):
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


def build_final_cnn_model(input_shape, forecast_period):
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape))

    for i in range(best_params['num_layers']):
        filters = best_params[f'filters_{i}']
        kernel_size = best_params[f'kernel_size_{i}']

        model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same'))
        model.add(BatchNormalization())

        pool_size = ceil(model.layers[-1].output_shape[1] / 2)
        if pool_size > 1:
            model.add(MaxPooling1D(pool_size=pool_size))
        else:
            model.add(MaxPooling1D(pool_size=1))

        model.add(Dropout(rate=best_params['dropout_rate']))

    model.add(Flatten())
    model.add(Dense(32, activation='relu', kernel_regularizer=l1(best_params['l1_regularizer']),
                    activity_regularizer=l2(best_params['l2_regularizer'])))
    model.add(Dense(forecast_period, activation='linear'))

    # Compile the model with the specified optimizer and loss function
    optimizer = c_optimizer(best_params['learning_rate'], best_params['decay_steps'], best_params['optimizer'])
    model.compile(loss='mean_absolute_error', optimizer=optimizer)

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
    # By masking the zeros, the model can learn to ignore the missing values and focus on the valid data.
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
    # By masking the zeros, the model can learn to ignore the missing values and focus on the valid data.
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
    # By masking the zeros, the model can learn to ignore the missing values and focus on the valid data.
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
    # By masking the zeros, the model can learn to ignore the missing values and focus on the valid data.
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
