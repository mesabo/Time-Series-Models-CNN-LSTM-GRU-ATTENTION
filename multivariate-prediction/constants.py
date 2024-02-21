#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 18:35:52 2024

@author: mesabo
"""

# Define model names as variables
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


# Define saving paths
SAVING_PATH = "../output/models/"
SAVING_METRICS_PATH = "../output/evaluation_metrics.json"
SAVING_LOSSES_PATH = "../output/models_losses.json"

# Define dataset paths
DATASET_PATH = "../input/household_power_consumption.txt"  # Adjust as needed


