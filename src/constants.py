#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 18:35:52 2024

@author: mesabo
"""

# Define model names as variables
EPOCH = 100
BATCH_SIZE = 64
SEEDER = 42
PARAMS_GRID = {
    'optimizer': ['adam', 'sgd', 'rmsprop'],
    'num_layers': [1, 2],
    'units': [100],
    'dropout_rate': [0.2, 0.3],
    'attention_units': [100],
    'filters': [32, 64, 128],
    'kernel_size': [3],
    'activation': ['relu', 'tanh', 'sigmoid']
}

# Simple models
LSTM_MODEL = "LSTM-based"
GRU_MODEL = "GRU-based"
CNN_MODEL = "CNN-Based"

# Bi models
BiLSTM_MODEL = "BiLSTM-based"
BiGRU_MODEL = "BiGRU-based"

# Simple models + Attention
LSTM_ATTENTION_MODEL = "LSTM-Attention-based"
GRU_ATTENTION_MODEL = "GRU-Attention-based"
CNN_ATTENTION_MODEL = "CNN-Attention-Based"

# Bi models + Attention
BiLSTM_ATTENTION_MODEL = "BiLSTM-Attention-based"
BiGRU_ATTENTION_MODEL = "BiGRU-Attention-based"

# Hybrid models
CNN_LSTM_MODEL = "CNN-LSTM-based"
CNN_GRU_MODEL = "CNN-GRU-based"
CNN_BiLSTM_MODEL = "CNN-BiLSTM-based"
CNN_BiGRU_MODEL = "CNN-BiGRU-based"
CNN_LSTM_ATTENTION_MODEL = "CNN-LSTM-Attention-based"
CNN_GRU_ATTENTION_MODEL = "CNN-GRU-Attention-based"
CNN_BiLSTM_ATTENTION_MODEL = "CNN-BiLSTM-Attention-based"
CNN_BiGRU_ATTENTION_MODEL = "CNN-BiGRU-Attention-based"

# Custom Hybrid models
CNN_LSTM_ATTENTION_LSTM_MODEL = "CNN-LSTM-Attention-LSTM-based"
CNN_GRU_ATTENTION_GRU_MODEL = "CNN-GRU-Attention-GRU-based"
CNN_BiLSTM_ATTENTION_BiLSTM_MODEL = "CNN-BiLSTM-Attention-BiLSTM-based"
CNN_BiGRU_ATTENTION_BiGRU_MODEL = "CNN-BiGRU-Attention-BiGRU-based"

# Custom Deep Hybrid models
CNN_ATTENTION_LSTM_MODEL = "CNN-Attention-LSTM-based"
CNN_ATTENTION_GRU_MODEL = "CNN-Attention-GRU-based"
CNN_ATTENTION_BiLSTM_MODEL = "CNN-Attention-BiLSTM-based"
CNN_ATTENTION_BiGRU_MODEL = "CNN-Attention-BiGRU-based"

# Custom Mode Deep Hybrid models
CNN_ATTENTION_LSTM_ATTENTION_MODEL = "CNN-Attention-LSTM-Attention-based"
CNN_ATTENTION_GRU_ATTENTION_MODEL = "CNN-Attention-GRU-Attention-based"
CNN_ATTENTION_BiLSTM_ATTENTION_MODEL = "CNN-Attention-BiLSTM-Attention-based"
CNN_ATTENTION_BiGRU_ATTENTION_MODEL = "CNN-Attention-BiGRU-Attention-based"


# Define saving paths
SAVING_MODEL_DIR = "../output/models/"
SAVING_METRIC_DIR = "../output/metrics/"
SAVING_PREDICTION_DIR = "../output/predictions/"
SAVING_LOSS_DIR = "../output/losses/"
SAVING_METRICS_PATH = "../output/metrics/evaluation_metrics.json"
SAVING_LOSSES_PATH = "../output/losses/models_losses.json"

# Define dataset paths
DATASET_PATH = "../input/household_power_consumption.txt"  # Adjust as needed

CHECK_PATH = "../output/checks/"
CHECK_HYPERBAND = "../output/hyperband/"
CHECK_HYPERBAND_PATH = "../output/hyperband/best_params.json"

