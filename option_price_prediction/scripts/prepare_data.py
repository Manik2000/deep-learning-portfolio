import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for training.
    """
    X = data.drop("y", axis=1)
    y = data["y"].to_numpy().reshape(-1, 1)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    return pd.DataFrame(np.concatenate([X, y], axis=1), columns=data.columns)


if __name__ == "__main__":
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    data = pd.read_parquet("../data/raw/train.parquet")
    data = prepare_data(data)
    data.to_parquet("../data/processed/train.parquet", index=False)

    data = pd.read_parquet("../data/raw/test.parquet")
    data = prepare_data(data)
    data.to_parquet("../data/processed/test.parquet", index=False)
