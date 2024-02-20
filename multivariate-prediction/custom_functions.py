#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 17:49:39 2024

@author: mesabo
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
import os

def evaluate_model(testY, testPredict):
    mse = mean_squared_error(testY, testPredict)
    mae = mean_absolute_error(testY, testPredict)
    rmse = mean_squared_error(testY, testPredict, squared=False)
    mape = np.mean(np.abs((testY - testPredict) / testY)) * 100
    print(f"[-----MODEL METRICS-----]\n")
    print(f"[-----MSE: {mse}-----]\n")
    print(f"[-----MAE: {mae}-----]\n")
    print(f"[-----RMSE: {rmse}-----]\n")
    print(f"[-----MAPE: {mape}-----]\n")
    return mse, mae, rmse, mape

def plot_losses(history):
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

def plot_evaluation_metrics(mse, mae, rmse, mape):
    metrics = ['MSE', 'MAE', 'RMSE']
    values = [mse, mae, rmse]
    
    plt.bar(metrics, values, color=['steelblue', 'limegreen', 'orangered'])
    plt.title('Evaluation Metrics')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.show()
    
def save_model(model, saving_path):
        model.save(saving_path + ".h5")
        
def save_evaluation_metrics(saving_path, model_type, mse, mae, rmse, mape):
    # Load existing data from the file or initialize an empty dictionary
    if os.path.exists(saving_path):
        with open(saving_path, 'r') as file:
            evaluation_data = json.load(file)
    else:
        evaluation_data = {}

    # Update or add metrics for the current model
    evaluation_data[model_type] = {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }

    # Save the updated data back to the file
    with open(saving_path, 'w') as file:
        json.dump(evaluation_data, file, indent=2)
