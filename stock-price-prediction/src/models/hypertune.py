from __future__ import absolute_import

import os

import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from src.data.lookback import create_lookback_dataset
from src.models.lstm_gru import MyGRU, MyLSTM

# sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))




if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))
    epochs = 20

    train = pd.read_csv(os.path.join("data", "processed", "train.csv"))
    validation_split = 0.2
    index = int(len(train) * (1 - validation_split))

    train_data, val_data = train[:index], train[index:]

    def set_up_torch_data(look_back):
        train_x, train_y = list(
            map(
                lambda x: torch.from_numpy(x),
                create_lookback_dataset(train_data, look_back),
            )
        )
        train = torch.utils.data.TensorDataset(train_x, train_y)
        train_loader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=False)
        return train_loader

    def lstm_objective(trial):
        look_back = trial.suggest_int("look_back", 1, 20)

        train_loader = set_up_torch_data(look_back)

        hidden_size = trial.suggest_int("hidden_size", 1, 20)
        num_layers = trial.suggest_int("num_layers", 1, 5)
        lstm = MyLSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers)

        lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(lstm.parameters(), lr=lr)

        for _ in range(epochs):
            loss_ = 0.0
            for x, y in enumerate(train_loader):
                optimizer.zero_grad()
                output = lstm(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                loss_ += loss.item()

        return loss_ / len(train_loader)

    def gru_objective(trial):
        look_back = trial.suggest_int("look_back", 1, 20)

        train_loader = set_up_torch_data(look_back)

        hidden_size = trial.suggest_int("hidden_size", 1, 20)
        num_layers = trial.suggest_int("num_layers", 1, 5)
        gru = MyGRU(input_size=1, hidden_size=hidden_size, num_layers=num_layers)

        lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(gru.parameters(), lr=lr)

        for _ in range(epochs):
            loss_ = 0.0
            for x, y in enumerate(train_loader):
                optimizer.zero_grad()
                output = gru(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                loss_ += loss.item()

        return loss_ / len(train_loader)

    print("LSTM hyperparameter tuning")
    study = optuna.create_study(direction="minimize")
    study.optimize(lstm_objective, n_trials=5)
    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)

    print("GRU hyperparameter tuning")
    study = optuna.create_study(direction="minimize")
    study.optimize(gru_objective, n_trials=5)
    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)
