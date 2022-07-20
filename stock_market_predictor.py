from dataclasses import dataclass
import os
from re import S
from tkinter import E
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler


class StockMarketPredictor:

    def __init__(self, data_file_path, epochs=100, batch_size=64):
        self.data_file_path = data_file_path
        self.data = None
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def prepare_data(self):
        # data reading
        data = pd.read_csv(self.data_file_path)
        del data['Date']

        # z-score normalization
        scaler = StandardScaler()
        trans_data = scaler.fit_transform(np.array(data))

        # Params
        seq_len = 10
        train_divide = 0.8
        val_divide = 0.1

        # Dataset Generation
        x = []
        y = []
        for i in range(len(trans_data)-seq_len):
            x.append(trans_data[i:i+seq_len, :])
            y.append(trans_data[i+seq_len, 4])
        x = np.array(x)
        y = np.array(y)

        # Data Split
        train_ind = int(len(trans_data)*train_divide)
        val_ind = train_ind + int(len(trans_data)*val_divide)
        x_train = x[:train_ind]
        y_train = y[:train_ind]
        x_val = x[train_ind:val_ind]
        y_val = y[train_ind:val_ind]
        x_test = x[val_ind:]
        y_test = y[val_ind:]
        self.data = {'x_train': x_train, 'y_train': y_train, 'x_val': x_val,
                     'y_val': y_val, 'x_test': x_test, 'y_test': y_test, 'scaler': scaler}

    def prepare_model(self):
        model = keras.models.Sequential()
        model.add(layers.Conv1D(32, 1, padding='same',
                                activation='tanh', input_shape=(10, 6)))
        model.add(layers.MaxPool1D(pool_size=1, padding='same'))
        model.add(layers.LSTM(64, activation='tanh'))
        model.add(layers.Dense(1))

        model.compile(loss='mean_absolute_error',
                      optimizer=keras.optimizers.Adam(learning_rate=0.001))
        self.model = model

    def train(self):
        if self.data == None:
            self.prepare_data()
        if self.model == None:
            self.prepare_model()
        self.model.fit(self.data['x_train'], self.data['y_train'], validation_data=(
            self.data['x_val'], self.data['y_val']), epochs=self.epochs, batch_size=self.batch_size)

    def test(self):
        if self.model == None:
            print("Run train first")
        self.model.evaluate(self.data['x_test'],
                            self.data['y_test'], verbose=True)

    def plot(self):
        scaler = self.data['scaler']
        mean_ = scaler.mean_[4]
        sd_ = np.sqrt(scaler.var_[4])
        trans_predicted_close = self.model(self.data['x_test'])
        prediction_close = sd_*trans_predicted_close + mean_
        actual_close = sd_*self.data['y_test'] + mean_
        name = self.data_file_path.split('/')[-1].split('.')[0]
        self.model.save('./Models/'+name)
        plt.plot(actual_close, color='red')
        plt.plot(prediction_close, color='blue')
        plt.xlabel('Time')
        plt.ylabel('Closing Stock Prices')
        plt.title(name)
        plt.legend(['Real Price', 'Predicted Price'])
        plt.savefig('./Results/'+name)
        plt.clf()


path = './Datasets'
data_file_paths = []
for root, dirs, files in os.walk(path):
    for name in files:
        if name.endswith((".csv")):
            data_file_path = path + '/' + name
            data_file_paths.append(data_file_path)

for data_file_path in data_file_paths:
    stock_market_predictor = StockMarketPredictor(
        data_file_path)
    stock_market_predictor.train()
    stock_market_predictor.test()
    stock_market_predictor.plot()
