import numpy as np


def arr_to_str(a: np.ndarray):
    """Returns ;-separated string representing the input"""
    return ";".join([str(x) for x in a.reshape(-1)])
