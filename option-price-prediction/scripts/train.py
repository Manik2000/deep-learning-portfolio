import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from model import OptionPricer
from torch.utils.data import DataLoader, TensorDataset

EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
HIDDEN_LAYERS = 2
UNITS = 80


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    model = OptionPricer(HIDDEN_LAYERS, UNITS)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train = pd.read_parquet("../data/processed/train.parquet")
    train_x, train_y = train.drop("y", axis=1), train["y"]

    test_x = pd.read_parquet("../data/processed/test.parquet")[
        ["S", "K", "T", "r", "sigma"]
    ]
    test_y = pd.read_parquet("../data/processed/test.parquet")["y"]

    train_x = torch.tensor(train_x.values, dtype=torch.float32)
    train_y = torch.tensor(train_y.values, dtype=torch.float32).reshape(-1, 1)

    test_x = torch.tensor(test_x.values, dtype=torch.float32)
    test_y = torch.tensor(test_y.values, dtype=torch.float32).reshape(-1, 1)

    dataloader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, num_workers=4
    )

    for epoch in range(EPOCHS):
        for i, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

    torch.save(model.state_dict(), os.path.join("..", "models", "nn.pt"))
