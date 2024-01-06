import torch.nn as nn


class OptionPricer(nn.Module):
    def __init__(self, hidden_layers, units, input_size=5):
        super(OptionPricer, self).__init__()
        self.units = units
        self.layers = []
        self.layers.append(nn.Linear(input_size, units))
        self.layers.append(nn.Dropout(0.15))
        self.layers.append(nn.ReLU())

        for _ in range(hidden_layers):
            self.layers.append(nn.Linear(units, units))
            self.layers.append(nn.Dropout(0.15))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(units, 1))
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.model(x)
        return x
