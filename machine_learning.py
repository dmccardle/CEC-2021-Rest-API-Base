import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from math import trunc
import os

scaler = MinMaxScaler(feature_range=(-1, 1))
window_size = 10

def load_datasets():
    years = [2015, 2016, 2017, 2018]

    dataset = None
    for year in years:
        file_path = f'PastYearData/NBTrend{year}.csv'
        yearly_dataset = pd.read_csv(file_path, header=None)
        if dataset is None:
            dataset = yearly_dataset.copy()
        else:
            dataset = dataset.append(yearly_dataset, ignore_index=True)

    test_dataset_size = 0
    dataset = scaler.fit_transform(dataset)
    if test_dataset_size != 0:
        train_dataset = dataset[:-test_dataset_size].astype(float)
        test_dataset = dataset[-test_dataset_size:].astype(float)
    else:
        train_dataset = dataset
        test_dataset = None
    return train_dataset, test_dataset


class RecurrentNN(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=10, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.dense = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        output, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 7, -1))
        predictions = self.dense(output)
        return predictions[-1]


def create_model_input(dataset, train_window=window_size):
    dataset = torch.FloatTensor(dataset)
    inout_seq = create_inout_sequences(dataset, train_window)
    return inout_seq


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


def train(train_data, model, epochs=2048, lr=0.001):
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(model)

    train_inout_seq = create_model_input(train_data)

    for i in range(epochs):
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))

            output = model(seq)

            mse_loss = loss_function(output, labels)
            mse_loss.backward()
            optimizer.step()

        if i % 16 == 1:
            print(f'Epoch #{i:3}\tMSE Loss: {mse_loss.item():.5f}')

    print(f'Epoch #{i:3}\tMSE Loss: {mse_loss.item():.5f}')


def test(test_data, model):
    model.eval()
    loss_function = nn.MSELoss()

    test_inout_seq = create_model_input(test_data)
    actual_predictions = []
    with torch.no_grad():
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))
        for seq, labels in test_inout_seq:
            output = model(seq)
            actual_predictions.append(scaler.inverse_transform(output.numpy().T))

            mse_loss = loss_function(output.T, labels)

        print(f'Test MSE Loss: {mse_loss.item():.5f}')
    return actual_predictions


def predict(input_seq):
    assert input_seq.shape == (window_size, 7)

    normalized_seq = scaler.transform(input_seq)
    rnn = torch.load('recurrent-neural-network.model')
    rnn.eval()
    with torch.no_grad():
        rnn.hidden_cell = (torch.zeros(1, 1, rnn.hidden_layer_size),
                           torch.zeros(1, 1, rnn.hidden_layer_size))
        output = rnn(torch.FloatTensor(normalized_seq))
    return scaler.inverse_transform(output.numpy().T)[0]


def forecast_through_2022():
    seq = scaler.inverse_transform(train_data[-window_size:])
    forecast = []
    for month in range(8 + 4 * 12):
        prediction = predict(seq)
        forecast.append([trunc(x*100)/100 for x in prediction])
        seq = np.append(seq, prediction.reshape(1, 7), axis=0)[1:]

    forecast_df = pd.DataFrame(forecast)
    for i in [1, 2, 3, 4]:
        if not os.path.exists(f'Forecast/WindowSize{window_size}'):
            os.makedirs(f'Forecast/WindowSize{window_size}')
        forecast_df[12*(i-1):12*i].to_csv(f'Forecast/WindowSize{window_size}/NBTrend{2018+i}_Forecast.csv', header=False, index=False)


if __name__ == '__main__':
    for i in range(8, 13):
        window_size = i
        train_data, test_data = load_datasets()
        model = RecurrentNN()
        train(train_data, model, epochs=1024, lr=0.0005)
        torch.save(model, 'recurrent-neural-network.model')
        forecast_through_2022()
    pass
