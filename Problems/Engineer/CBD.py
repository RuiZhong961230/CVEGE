"""
Hadi Bayzidi, Siamak Talatahari, Meysam Saraee, et al.
Social Network Search for Solving Engineering Optimization Problems[J].
Computational Intelligence and Neuroscience
"""
import numpy as np


def CBD_obj(X):
    """
    Corrugated Bulkhead Design
    :param X:
    0 <= x1, x2, x3 <= 100,
    0 <= x4 <= 5
    :return:
    """
    return 5.885 * X[3] * (X[0] + X[2]) / (X[0] + np.sqrt(abs(X[2] ** 2 - X[1] ** 2)))


def CBD_cons(X):
    con1 = -X[3] * X[1] * (0.4 * X[0] + X[2] / 6) + 8.94 * (X[0] + np.sqrt(abs(X[2] ** 2 - X[1] ** 2)))
    con2 = -X[3] * X[1] ** 2 * (0.2 * X[0] + X[2] / 12) + 2.2 * (8.94 * (X[0] + np.sqrt(abs(X[2] ** 2 - X[1] ** 2)))) ** (4 / 3)
    con3 = -X[3] + 0.0156 * X[0] + 0.15
    con4 = -X[3] + 0.0156 * X[2] + 0.15
    con5 = -X[3] + 1.05
    con6 = -X[2] + X[1]
    return [con1, con2, con3, con4, con5, con6]
