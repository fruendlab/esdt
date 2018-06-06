import numpy as np


def safe_log(p):
    return np.log(np.clip(p, 1e-7, 1e7))
