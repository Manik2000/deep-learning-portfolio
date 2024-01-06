import os

import numpy as np
from scipy.stats import expon, norm, uniform


def generate_distribution_sample(sample_size, distribution, *args, **kwargs):
    dist = distribution(*args, **kwargs)
    return dist.rvs(sample_size)


def save_sample(sample, filename):
    np.savetxt(filename, sample)


if __name__ == "__main__":
    np.random.seed(30_08_2000)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    sample_size = 100_000

    distributions = [norm, uniform, expon]
    distributions_args = [{"loc": 0, "scale": 1}, {"loc": 0, "scale": 1}, {"scale": 1}]
    distributions_names = ["normal", "uniform", "exponential"]

    for dist, dist_args, dist_name in zip(
        distributions, distributions_args, distributions_names
    ):
        sample = generate_distribution_sample(sample_size, dist, **dist_args)
        save_sample(sample, os.path.join("..", "data", f"{dist_name}_sample.txt"))
