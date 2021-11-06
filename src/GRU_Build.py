from matplotlib import pyplot as plt
import os

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import tensorflow as tf
from data_prep import normalize_data
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, GRU, LSTM, Activation
from tensorflow.keras.optimizers import SGD, Adam
from keras.layers import Dropout
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import pandas as pd

def GRU_builder(X_train, y_train, X_test, y_test, scaler, scaler1):
    model = Sequential()

    # GRU layers
    model.add(GRU(1028, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), activation='tanh'))
    model.add(Dropout(0.2))
    model.add(GRU(512, input_shape=(X_train.shape[1], X_train.shape[2]), activation='tanh'))
    model.add(Dropout(0.2))

    # Dense Layer
    model.add(Dense(2))
    model.add(Activation('linear'))

    # Print model summary, compile it, and fit it.
    model.summary()
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error')
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=50)

    # Get predictions and scale them back to price
    predictions_train, predictions_test = model.predict(X_train), model.predict(X_test)
    scaler = MinMaxScaler(feature_range=(scaler.data_min_[0], scaler.data_max_[0]))
    predictions_train = scaler.fit_transform(predictions_train)

    # scaler1 = MinMaxScaler(feature_range=(scaler1.data_min_[0], scaler1.data_max_[0]))
    predictions_test = scaler.transform(predictions_test)

    return model, predictions_train, predictions_test


# Create the model and predict on test/validation set.
X_train, y_train, X_test, y_test, scaler, scaler1 = normalize_data('../data/data.csv', 20, 1)
model, predictions_train, predictions_test = GRU_builder(X_train, y_train, X_test, y_test, scaler, scaler1)

# Get train/test data
data = pd.read_csv('../data/data.csv').round(2)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data_train = data['Close']['1995':'2020']
data_test = data['Close']['2021':]

# Prepare train data for plotting
pred_plot_train = pd.DataFrame(columns=['Close', 'Prediction'])
pred_plot_train['Close'] = data_train[X_train.shape[1]+y_train.shape[1]-1:]
pred_plot_train['Prediction'] = predictions_train[0:, 0]

# Plot train data
plt.plot(pred_plot_train[['Close', 'Prediction']])
plt.show()

# Prepare test data for plotting
pred_plot_test = pd.DataFrame(columns=['Close', 'Prediction'])
pred_plot_test['Close'] = data_test
pred_plot_test['Prediction'] = predictions_test[0:, 0]

# Plot test data
plt.plot(pred_plot_test[['Close', 'Prediction']])
plt.show()
