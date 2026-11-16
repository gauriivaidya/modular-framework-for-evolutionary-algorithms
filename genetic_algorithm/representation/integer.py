import numpy as np

def integer(size, low=0, high=None, **kwargs):
    if high is None:
        high = size - 1
    return np.random.randint(low, high + 1, size, dtype=int)