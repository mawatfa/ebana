import numpy as np

def softmax(arr):
    eps = np.finfo(float).eps
    exp_arr = np.exp(arr)
    return exp_arr / (np.sum(exp_arr) + eps)
