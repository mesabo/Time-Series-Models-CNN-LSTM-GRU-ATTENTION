#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 15:22:58 2024

@author: mesabo
"""
# Import Packages
from numpy import array, split
from pandas import read_csv
from math import sqrt
from sklearn.metrics import mean_squared_error


class ModelEvaluation:
    # evaluate one or more weekly forecasts against expected values
    def evaluate_forecasts(actual, predicted):
    	scores = list()
    	# calculate an RMSE score for each day
    	for i in range(actual.shape[1]):
    		# calculate mse
    		mse = mean_squared_error(actual[:, i], predicted[:, i])
    		# calculate rmse
    		rmse = sqrt(mse)
    		# store
    		scores.append(rmse)
    	# calculate overall RMSE
    	s = 0
    	for row in range(actual.shape[0]):
    		for col in range(actual.shape[1]):
    			s += (actual[row, col] - predicted[row, col])**2
    	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    	return score, scores
    
    # summarize scores
    def summarize_scores(name, score, scores):
    	s_scores = ', '.join(['%.1f' % s for s in scores])
    	print('%s: [%.3f] %s' % (name, score, s_scores))


class DataProcessing:
    # split a univariate dataset into train/test sets
    def split_dataset(data):
        # split into standard weeks
    	train, test = data[1:-328], data[-328:-6]
    	# restructure into windows of weekly data
    	train = array(split(train, len(train)/7))
    	test = array(split(test, len(test)/7))
    	return train, test

    def read_dataset(path='../input/household_power_consumption_days.csv'):
        dataset = read_csv(path, header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
        print(f"Dataset shape: {dataset.shape}")
        print(f"Dataset column: {dataset.columns}")
        return dataset
    
    # convert history into inputs and outputs
    def to_supervised(train, n_input, n_out=7):
    	# flatten data (159, 7, 8) ==> (159*7=1113, 8): 3D-->2D
    	data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    	X, y = [], []
    	in_start = 0
    	# step over the entire history one time step at a time
    	for _ in range(len(data)):
    		# define the end of the input sequence
    		in_end = in_start + n_input
    		out_end = in_end + n_out
    		# ensure we have enough data for this instance
    		if out_end <= len(data):
    			x_input = data[in_start:in_end, 0]
    			x_input = x_input.reshape((len(x_input), 1))
    			X.append(x_input)
    			y.append(data[in_end:out_end, 0])
    		# move along one time step
    		in_start += 1
    	return array(X), array(y)

