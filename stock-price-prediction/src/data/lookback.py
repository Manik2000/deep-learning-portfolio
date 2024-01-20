import numpy as np


def create_lookback_dataset(data, look_back=1):
    data_x, data_y = [], []
    for i in range(len(data) - look_back - 1):
        data_x.append(data[i:(i + look_back)])
        data_y.append(data[i + look_back])
    return np.array(data_x), np.array(data_y)
