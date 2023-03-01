import numpy as np


def smape(y_true, y_pred):
    num = np.abs(y_true - y_pred)
    den = (np.abs(y_true) + np.abs(y_pred))/2
    num[den == 0] = 0
    den[den == 0] = 1
    return np.mean(num/den) * 100
