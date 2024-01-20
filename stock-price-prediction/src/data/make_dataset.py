import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler



if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__)))
    print(os.getcwd())
    df = pd.read_csv(os.path.join('..', '..', 'data', 'raw', 'MSFT.csv'))

    open_price = df["Open"].values

    test_size = int(len(open_price) * 0.2)
    train_size = len(open_price) - test_size

    train, test = open_price[0:train_size].reshape(-1, 1), open_price[train_size:len(open_price)].reshape(-1, 1)

    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    train = pd.DataFrame(train)
    test = pd.DataFrame(test)

    train.to_csv(os.path.join('..', '..', 'data', 'processed', 'train.csv'))
    test.to_csv(os.path.join('..', '..', 'data', 'processed', 'test.csv'))
