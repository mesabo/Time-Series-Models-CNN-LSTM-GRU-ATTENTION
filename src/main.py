#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 17:48:44 2024

@author: mesabo
"""

import time
import numpy as np
import tensorflow as tf
from data_processing import preprocess_and_split_dataset
from custom_models import (build_model, train_model, make_predictions)
from custom_functions import (evaluate_model, plot_losses, plot_evaluation_metrics,
                              save_evaluation_metrics, save_loss_to_txt,
                              predict_next_x_days, save_trained_model,
                              load_trained_model, plot_predictions, save_best_params)
from hyperparameter_tuning import (tune_custom_model)
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
    SAVING_MODEL_DIR, SAVING_METRIC_DIR, SAVING_LOSS_DIR, BASE_PATH,
    SAVING_PREDICTION_DIR, SAVING_METRICS_PATH, SAVING_LOSSES_PATH, SEEDER,
    HYPERBAND_PATH, DATASET_FEATURES_PATH, ELECTRICITY_DATASET_PATH,
    ELECTRICITY, WATER, WIND, GOLD
)
import warnings

warnings.filterwarnings("ignore")


def run_model(input_shape, forecast_period, trainX, trainY, testX, testY, scaler, model_type, series_type):
    print(f'**************************************************************')
    print(f'**                {model_type}                **')
    print(f'**************************************************************')
    model_name = SAVING_MODEL_DIR + model_type + '.keras'
    saving_path = BASE_PATH + series_type + '/'
    print(f"BASE PATH 📌📌📌  {saving_path}  📌📌📌")
    model_lstm_based = build_model(len(trainX), model_type, input_shape, forecast_period)
    model_lstm_based.summary()

    history = train_model(model_lstm_based, trainX, trainY, testX, testY)

    testPredict, testOutput = make_predictions(model_lstm_based, testX, testY, scaler)

    plot_predictions(testPredict, testOutput, model_type, saving_path + SAVING_PREDICTION_DIR)

    plot_losses(history, model_type, saving_path + SAVING_LOSS_DIR)

    mse, mae, rmse, mape = evaluate_model(testY, testPredict)
    plot_evaluation_metrics(mse, mae, rmse, mape, model_type, saving_path + SAVING_METRIC_DIR)

    save_evaluation_metrics(saving_path + SAVING_METRICS_PATH, model_type, mse, mae, rmse, mape)
    save_loss_to_txt(saving_path + SAVING_LOSSES_PATH, model_type, history)
    save_trained_model(model_lstm_based, saving_path + model_name)

    return model_lstm_based


def tune_models(look_backs, forecast_periods, model_types, series_type):
    for _ser in series_type:
        for look_back_day in look_backs:
            for forecast_day in forecast_periods:
                print(f"Tuning with look_back={look_back_day} and forecast_period={forecast_day}")
                trainX, trainY, testX, testY, scaler = preprocess_and_split_dataset(_ser, "D",
                                                                                    look_back_day, forecast_day)
                dataX = (trainX, trainY, testX, testY)
                input_shape = trainX.shape[-2:]
                band = f"{BASE_PATH + _ser}/{HYPERBAND_PATH}{look_back_day}_{forecast_day}"
                path = f"{BASE_PATH + _ser}/{HYPERBAND_PATH}{look_back_day}_{forecast_day}_best_params.json"
                for model in model_types:
                    start_time = time.time()
                    best_params = tune_custom_model(dataX, input_shape, forecast_day, model, band)
                    end_time = time.time()
                    total_time = end_time - start_time
                    save_best_params(path, model, best_params, total_time)
                    print(
                        f"----------BEST PARAMS FOR LB_DAYS={look_back_day} and FP_DAY={forecast_day}----------\n{best_params.values}")


def main():
    warnings.filterwarnings("ignore")
    # Set random seed for TensorFlow
    tf.random.set_seed(SEEDER)
    # Set random seed for NumPy
    np.random.seed(SEEDER)
    series = [ELECTRICITY, ]
    model_types = [
        LSTM_MODEL, GRU_MODEL, CNN_MODEL, BiLSTM_MODEL, BiGRU_MODEL,
        CNN_LSTM_MODEL, CNN_GRU_MODEL, CNN_BiLSTM_MODEL, CNN_BiGRU_MODEL,
        LSTM_ATTENTION_MODEL, GRU_ATTENTION_MODEL, CNN_ATTENTION_MODEL,
        BiLSTM_ATTENTION_MODEL, BiGRU_ATTENTION_MODEL,
        CNN_LSTM_ATTENTION_MODEL, CNN_GRU_ATTENTION_MODEL,
        CNN_BiLSTM_ATTENTION_MODEL, CNN_BiGRU_ATTENTION_MODEL,
        CNN_ATTENTION_LSTM_MODEL, CNN_ATTENTION_GRU_MODEL,
        CNN_ATTENTION_BiLSTM_MODEL, CNN_ATTENTION_BiGRU_MODEL,
        CNN_ATTENTION_LSTM_ATTENTION_MODEL, CNN_ATTENTION_GRU_ATTENTION_MODEL,
        CNN_ATTENTION_BiLSTM_ATTENTION_MODEL, CNN_ATTENTION_BiGRU_ATTENTION_MODEL, ]

    # Given look_back days observations, forecast next forecast_period observations
    look_backs = [7, 10, 14, 30]
    forecast_periods = [1, 2, 3, 6, 7]

    # Preprocess and split dataset
    trainX, trainY, testX, testY, scaler = preprocess_and_split_dataset(ELECTRICITY, "D", look_backs[0],
                                                                        forecast_periods[1])
    input_shape = trainX.shape[-2:]
    '''
    for model in model_types:  # [CNN_MODEL]
        final_model = run_model(input_shape, forecast_periods[1], trainX, trainY, testX, testY, scaler, model,
                                ELECTRICITY)
    '''

    # testPredict, testOutput = make_predictions(final_model, valX, valY, scaler)
    # plot_predictions(testPredict, testOutput, CNN_MODEL, BASE_PATH + ELECTRICITY + '/' + SAVING_PREDICTION_DIR)

    ##### HYPER PARAMETER TUNING  ALL THE MODELS
    tune_models(look_backs, forecast_periods, model_types, series)


if __name__ == "__main__":
    main()
