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
from custom_models import (build_model, train_model, make_predictions,
                           tune_hyperparameters)
from custom_functions import (evaluate_model, plot_losses, plot_evaluation_metrics,
                              save_model, save_evaluation_metrics,save_loss_to_txt,
                              predict_next_x_days,save_trained_model,
                              load_trained_model, plot_predictions, save_best_params)
from hyperparameter_tuning import (build_lstm_model, tune_custom_model)
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
    SAVING_MODEL_DIR, SAVING_METRIC_DIR, SAVING_LOSS_DIR,
    SAVING_PREDICTION_DIR, SAVING_METRICS_PATH, SAVING_LOSSES_PATH, SEEDER,
    CHECK_HYPERBAND_PATH, DATASET_FEATURES_PATH, ELECTRICITY_DATASET_PATH,
    ELECTRICITY, WATER, WIND, GOLD
)
import warnings

warnings.filterwarnings("ignore")
def run_model(input_shape, forecast_period, trainX, trainY, valX, valY, testX, testY, scaler, model_type):
    print(f'**************************************************************')
    print(f'**                {model_type}                **')
    print(f'**************************************************************')
    model_name = SAVING_MODEL_DIR + model_type+'.keras'
    
    model_lstm_based = build_model(len(trainX), model_type, input_shape, forecast_period)
    model_lstm_based.summary()
    
    history = train_model(model_lstm_based, trainX, trainY, valX, valY)

    testPredict, testOutput = make_predictions(model_lstm_based, testX, testY, scaler)

    plot_predictions(testPredict, testOutput, model_type, SAVING_PREDICTION_DIR)
    
    plot_losses(history, model_type, SAVING_LOSS_DIR)

    mse, mae, rmse, mape = evaluate_model(testY, testPredict)
    plot_evaluation_metrics(mse, mae, rmse, mape, model_type, SAVING_METRIC_DIR)

    save_evaluation_metrics(SAVING_METRICS_PATH, model_type, mse, mae, rmse, mape)
    save_loss_to_txt(SAVING_LOSSES_PATH, model_type, history)
    save_trained_model(model_lstm_based, model_name)
    
    return model_lstm_based


def main():
    warnings.filterwarnings("ignore")
    # Set random seed for TensorFlow
    tf.random.set_seed(SEEDER)
    # Set random seed for NumPy
    np.random.seed(SEEDER)

    # Given look_back days observations, forecast next forecast_period observations
    look_back = 7
    forecast_period = 2

    # Preprocess and split dataset
    trainX, trainY, valX, valY, testX, testY, scaler = preprocess_and_split_dataset(ELECTRICITY, "D", look_back, forecast_period)
    dataX = (trainX, trainY, valX, valY)

    input_shape = trainX.shape[-2:]

    '''
    final_model = run_model(input_shape, forecast_period, trainX, trainY, valX, valY, testX,testY, scaler, CNN_MODEL)
    testPredict, testOutput = make_predictions(final_model, valX, valY, scaler)

    plot_predictions(testPredict, testOutput, CNN_MODEL, SAVING_PREDICTION_DIR)
    '''
    ##### HYPER PARAMETER TUNING  ALL THE MODELS
    '''
    model_types = [
        LSTM_MODEL, GRU_MODEL , CNN_MODEL, BiLSTM_MODEL , BiGRU_MODEL,
        LSTM_ATTENTION_MODEL, GRU_ATTENTION_MODEL , CNN_ATTENTION_MODEL , 
        BiLSTM_ATTENTION_MODEL , BiGRU_ATTENTION_MODEL , 
        CNN_LSTM_MODEL, CNN_GRU_MODEL, CNN_BiLSTM_MODEL, CNN_BiGRU_MODEL,
        CNN_LSTM_ATTENTION_MODEL , CNN_GRU_ATTENTION_MODEL , 
        CNN_BiLSTM_ATTENTION_MODEL , CNN_BiGRU_ATTENTION_MODEL , 
        CNN_ATTENTION_LSTM_MODEL , CNN_ATTENTION_GRU_MODEL , 
        CNN_ATTENTION_BiLSTM_MODEL , CNN_ATTENTION_BiGRU_MODEL,
        CNN_ATTENTION_LSTM_ATTENTION_MODEL,CNN_ATTENTION_GRU_ATTENTION_MODEL, 
        CNN_ATTENTION_BiLSTM_ATTENTION_MODEL, CNN_ATTENTION_BiGRU_ATTENTION_MODEL,]
    
    
    
    for model in model_types:
        start_time = time.time()
        best_params = tune_custom_model(dataX, input_shape, forecast_period, model)
        end_time = time.time()
        total_time = end_time - start_time
        save_best_params(CHECK_HYPERBAND_PATH, model, best_params, total_time)
        print(f"----------BEST PARAMS----------\n{best_params.values}")
    '''    


    ##### Run all models at a time
    model_types = [
        LSTM_MODEL, GRU_MODEL , CNN_MODEL, BiLSTM_MODEL , BiGRU_MODEL,
        LSTM_ATTENTION_MODEL, GRU_ATTENTION_MODEL , CNN_ATTENTION_MODEL , 
        BiLSTM_ATTENTION_MODEL , BiGRU_ATTENTION_MODEL , 
        CNN_LSTM_MODEL, CNN_GRU_MODEL, CNN_BiLSTM_MODEL, CNN_BiGRU_MODEL,
        CNN_LSTM_ATTENTION_MODEL , CNN_GRU_ATTENTION_MODEL , 
        CNN_BiLSTM_ATTENTION_MODEL , CNN_BiGRU_ATTENTION_MODEL , 
        CNN_ATTENTION_LSTM_MODEL , CNN_ATTENTION_GRU_MODEL , 
        CNN_ATTENTION_BiLSTM_MODEL , CNN_ATTENTION_BiGRU_MODEL,
        CNN_ATTENTION_LSTM_ATTENTION_MODEL,CNN_ATTENTION_GRU_ATTENTION_MODEL, 
        CNN_ATTENTION_BiLSTM_ATTENTION_MODEL, CNN_ATTENTION_BiGRU_ATTENTION_MODEL,]
    
    for model in model_types: 
        run_model(input_shape, forecast_period, trainX, trainY, valX, valY, testX, testY, scaler, model)


if __name__ == "__main__":
    main()
