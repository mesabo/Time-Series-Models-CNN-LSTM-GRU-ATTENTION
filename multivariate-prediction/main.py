#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 17:48:44 2024

@author: mesabo
"""

from data_processing import preprocess_and_split_dataset
from custom_models import build_model, train_model, make_predictions
from custom_functions import (evaluate_model, plot_losses, plot_evaluation_metrics, save_model, save_evaluation_metrics)
from constants import (
    LSTM_MODEL, CNN_MODEL, CNN_LSTM_MODEL, CNN_GRU_MODEL, CNN_BiLSTM_MODEL,
    CNN_BiGRU_MODEL, CNN_LSTM_AE_MODEL, CNN_BiLSTM_AE_MODEL, LSTM_ATTENTION_MODEL,
    CNN_ATTENTION_MODEL, CNN_LSTM_ATTENTION_MODEL, CNN_GRU_ATTENTION_MODEL,
    CNN_BiLSTM_ATTENTION_MODEL, CNN_BiGRU_ATTENTION_MODEL, CNN_LSTM_AE_ATTENTION_MODEL,
    CNN_BiLSTM_AE_ATTENTION_MODEL, SAVING_PATH, DATASET_PATH, SAVING_METRICS_PATH
)

def run_model(look_back, forecast_period, trainX, trainY, valX, valY, testX, testY, scaler, model_type):
    
    input_shape = (look_back, 1)
    # Build model
    model_lstm_based = build_model(len(trainX), model_type, input_shape, forecast_period)

    # Train model
    lstm_history = train_model(model_lstm_based, trainX, trainY, valX, valY)

    # Make prediction
    lstm_testPredict = make_predictions(model_lstm_based, testX, scaler)
    
    # Plot loss history
    plot_losses(lstm_history)

    # Evaluate model
    mse, mae, rmse, mape = evaluate_model(testY, lstm_testPredict)
    # Plot evaluation metrics
    plot_evaluation_metrics(mse, mae, rmse, mape)

    # Save evaluation metrics to JSON file
    save_evaluation_metrics(SAVING_METRICS_PATH, model_type, mse, mae, rmse, mape)
    
    
    
def main():
    # Define dataset path
    dataset_path = DATASET_PATH if DATASET_PATH else "../input/household_power_consumption.csv"

    # Define look back period and forecast period
    look_back = 10
    forecast_period = 7  # Change this as needed
    input_shape = (look_back, 1)  # Adjust according to your data

    # Preprocess and split dataset
    trainX, trainY, valX, valY, testX, testY, scaler = preprocess_and_split_dataset(dataset_path, look_back, forecast_period)
    
    
    run_model(look_back, forecast_period, trainX, trainY, valX, valY, testX, testY, scaler, CNN_BiLSTM_MODEL)
    
"""
    # Define model type    
    models = [CNN_GRU_MODEL, CNN_LSTM_MODEL, LSTM_MODEL]  # List of models to loop through
    for model_type in models:
        run_model(look_back, forecast_period, trainX, trainY, valX, valY, testX, testY, scaler, model_type)
"""
   


if __name__ == "__main__":
    main()
