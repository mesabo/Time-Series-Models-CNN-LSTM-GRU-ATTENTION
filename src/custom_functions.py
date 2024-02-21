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
from keras.models import save_model, load_model

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


def predict_next_x_days(model, X_new, days=7):
    predictions = []

    # Iterate over the next days
    for i in range(days):
        prediction = model.predict(X_new)
        
        predictions.append(prediction)
        # Update X_new for the next iteration
        # Shift the values by one day and append the new prediction
        X_new = np.roll(X_new, -1, axis=1)
        X_new[-1] = prediction[0] 

    predictions = np.array(predictions)
    
    return predictions

def plot_losses(history, model, save_path=None):
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{model} - Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        if save_path:
            file_name = f'{model}_evaluation_metrics.png'
            file_path = os.path.join(save_path, file_name)
            plt.savefig(file_path)
        
        plt.show()

def plot_evaluation_metrics(mse, mae, rmse, mape, model, save_path=None):
    metrics = ['MSE', 'MAE', 'RMSE']
    values = [mse, mae, rmse]
    
    plt.bar(metrics, values, color=['steelblue', 'limegreen', 'orangered'])
    plt.title(f'{model} - Evaluation Metrics')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    if save_path:
        file_name = f'{model}_evaluation_metrics.png'
        file_path = os.path.join(save_path, file_name)
        plt.savefig(file_path)
    
    plt.show()
    
    
def save_evaluation_metrics(saving_path, model_type, mse, mae, rmse, mape):
    if os.path.exists(saving_path):
        with open(saving_path, 'r') as file:
            evaluation_data = json.load(file)
    else:
        evaluation_data = {}

    evaluation_data[model_type] = {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }

    # Save the updated data back to the file
    with open(saving_path, 'w') as file:
        json.dump(evaluation_data, file, indent=2)

def save_loss_to_txt(saving_path, model_type, history):
    if os.path.exists(saving_path):
        with open(saving_path, 'r') as file:
            loss_data = json.load(file)
    else:
        loss_data = {}

    loss_data[model_type] = {
        'training_loss': history.history['loss'],
        'validation_loss': history.history['val_loss']
    }

    with open(saving_path, 'w') as file:
        json.dump(loss_data, file, indent=2)

def save_trained_model(model, path):
    save_model(model, path)

def load_trained_model(path):
    loaded_model = load_model(path)
    return loaded_model
