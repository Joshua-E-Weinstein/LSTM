from matplotlib import pyplot as plt
import os

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import tensorflow as tf
from data_prep import normalize_data
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, GRU, LSTM, Activation
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import BinaryCrossentropy
from keras.layers import Dropout, Bidirectional
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import io
import urllib
import base64

import pandas as pd

data_file = '../data/Min_Temp_Daily.csv'
feature_to_predict = 'Temp'
dates = ('1981', '1989', '1990')

def direction_loss(y_true, y_pred):
    # Future Prices
    y_next = y_true[1:]
    y_next = y_pred[1:]

    # Current Prices
    y_cur = y_true[:-1]
    y_cur = y_pred[:-1]

def GRU_builder(X_train, y_train, X_test, y_test, scaler, scaler1):
    model = Sequential()

    # GRU layers
    model.add(GRU(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), activation='tanh')) # tanh for both
    model.add(Dropout(0.2))
    model.add(GRU(32, input_shape=(X_train.shape[1], X_train.shape[2]), activation='tanh')) # tanh for both
    model.add(Dropout(0.2))

    # Dense Layer
    # model.add(Dense(32, activation='relu'))
    model.add(Dense(y_train.shape[1]))
    model.add(Activation('linear'))  # linear for price, sigmoid for growth

    # Print model summary, compile it, and fit it.
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error', metrics=['accuracy']) # mean_squared_error for price, binary_crossentropy for growth
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)
    model.summary()

    # Get predictions and scale them back to price
    print(X_train)
    print(y_train)
    predictions_train, predictions_test = model.predict(X_train), model.predict(X_test)
    scaler = MinMaxScaler(feature_range=(scaler.data_min_[0], scaler.data_max_[0]))
    predictions_train = scaler.fit_transform(predictions_train)

    # scaler1 = MinMaxScaler(feature_range=(scaler1.data_min_[0], scaler1.data_max_[0]))
    predictions_test = scaler.transform(predictions_test)

    return model, predictions_train, predictions_test

def plotting():
    # Create the model and predict on test/validation set.
    X_train, y_train, X_test, y_test, scaler, scaler1 = normalize_data(data_file, 2, 2, dates)
    model, predictions_train, predictions_test = GRU_builder(X_train, y_train, X_test, y_test, scaler, scaler1)

    # Get train/test data
    data = pd.read_csv(data_file).round(2)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data_train = data[feature_to_predict][dates[0]:dates[1]]
    data_test = data[feature_to_predict][dates[2]]

    # Prepare train data for plotting
    pred_plot_train = pd.DataFrame(columns=[feature_to_predict, 'Prediction'])
    pred_plot_train[feature_to_predict] = data_train[X_train.shape[1]+y_train.shape[1]-1:]
    pred_plot_train['Prediction'] = predictions_train[0:, 0]

    # Plot train data
    plt.plot(pred_plot_train[[feature_to_predict, 'Prediction']])
    plt.title("Train Data")
    plt.show()

    # Prepare test data for plotting
    pred_plot_test = pd.DataFrame(columns=[feature_to_predict, 'Prediction'])
    pred_plot_test[feature_to_predict] = data_test
    pred_plot_test['Prediction'] = predictions_test[0:, 0]

    # Plot test data
    plt.plot(pred_plot_test[[feature_to_predict, 'Prediction']])
    plt.title("Test Data")
    plt.show()
