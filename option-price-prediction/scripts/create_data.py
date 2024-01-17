import os

import numpy as np
import pandas as pd
from scipy.stats import norm, uniform


def black_scholes_call_price(S, K, T, r, sigma):
    """
    Black-Scholes call price.
    """
    d1 = (np.log(S / K) + (r + sigma * sigma / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


if __name__ == "__main__":
    random_seed = 30_08_2000
    np.random.seed(random_seed)
    os.chdir(os.path.abspath(os.path.dirname(__file__)))

    # Parameters
    dataset_size = 300_000
    S = uniform(10, 500).rvs(size=dataset_size)
    K = uniform(5, 760).rvs(size=dataset_size)
    T = uniform(1 / 220, 3).rvs(size=dataset_size)
    r = uniform(0.01, 0.05).rvs(size=dataset_size)
    sigma = uniform(0.05, 0.9).rvs(size=dataset_size)

    y = black_scholes_call_price(S, K, T, r, sigma)

    dataset = pd.DataFrame({"S": S, "K": K, "T": T, "r": r, "sigma": sigma, "y": y})

    train = dataset.sample(frac=0.8, random_state=random_seed)
    test = dataset.drop(train.index)

    train.to_parquet("../data/raw/train.parquet", index=False)
    test.to_parquet("../data/raw/test.parquet", index=False)
