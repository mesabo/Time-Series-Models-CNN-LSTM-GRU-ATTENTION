#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 18:35:52 2024

@author: mesabo
"""

# Define model names as variables
LSTM_MODEL = "LSTM-based"
CNN_MODEL = "CNN-Based"
CNN_LSTM_MODEL = "CNN-LSTM-based"
CNN_GRU_MODEL = "CNN-GRU-based"
CNN_BiLSTM_MODEL = "CNN-BiLSTM-based"
CNN_BiGRU_MODEL = "CNN-BiGRU-based"
CNN_LSTM_AE_MODEL = "CNN-LSTM-AutoEncoder-based"
CNN_BiLSTM_AE_MODEL = "CNN-BiLSTM-AutoEncoder-based"
LSTM_ATTENTION_MODEL = "LSTM-Attention-based"
CNN_ATTENTION_MODEL = "CNN-Attention-Based"
CNN_LSTM_ATTENTION_MODEL = "CNN-LSTM-Attention-based"
CNN_GRU_ATTENTION_MODEL = "CNN-GRU-Attention-based"
CNN_BiLSTM_ATTENTION_MODEL = "CNN-BiLSTM-Attention-based"
CNN_BiGRU_ATTENTION_MODEL = "CNN-BiGRU-Attention-based"
CNN_LSTM_AE_ATTENTION_MODEL = "CNN-LSTM-AutoEncoder-Attention-based"
CNN_BiLSTM_AE_ATTENTION_MODEL = "CNN-BiLSTM-AutoEncoder-Attention-based"

# Define saving paths
SAVING_PATH = "../output/models/"  # Adjust as needed
SAVING_METRICS_PATH = "../output/evaluation_metrics.json"

# Define dataset paths
DATASET_PATH = "../input/household_power_consumption.txt"  # Adjust as needed
