from sklearn.metrics import f1_score
import numpy as np


def get_optimal_threshold(y_true, pred_prob):
    max_f1, max_t = 0.0, 0.0
    for t in np.linspace(0, 1, num=100):
        f1 = f1_score(y_true, pred_prob >= t)
        if f1 >= max_f1:
            max_f1 = f1
            max_t = t
    return max_t
