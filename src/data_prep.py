import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# **Important** Make sure the first feature is the one being predicted #

def normalize_data(data_path, n_timesteps, for_periods, dates):
    # Read data and format dates into pandas recognizable format
    data = pd.read_csv(data_path).round(2)
    data['Date'] = pd.to_datetime(data['Date'])

    data.set_index('Date', inplace=True)  # Set dataframe index to date

    # Split data into train/test sets
    data_train = data[dates[0]:dates[1]]
    data_test = data.values[len(data)-len(data[dates[2]:])-n_timesteps-for_periods+1:]

    # Save length of train/test set
    data_train_len = len(data_train)
    data_test_len = len(data_test)

    # Normalize data between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_train_scaled = scaler.fit_transform(data_train)
    scaler1 = MinMaxScaler(feature_range=(0, 1))
    data_test_scaled = scaler.transform(data_test)

    # Split data into features and labels.
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for i in range(n_timesteps, data_train_len-for_periods+1):  # Split training data
        X_train.append(data_train_scaled[i-n_timesteps:i, 0:1])
        y_train.append(data_train_scaled[i:i+for_periods, 0])
    for i in range(n_timesteps, data_test_len-for_periods+1):  # Split testing data
        X_test.append(data_test_scaled[i-n_timesteps:i, 0:1])
        y_test.append(data_train_scaled[i:i+for_periods, 0])

    # Fix formatting so that the data doesn't contain numpy arrays
    X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

    return X_train, y_train, X_test, y_test, scaler, scaler1
