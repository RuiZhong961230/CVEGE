"""
Seyedali Mirjalili, Seyed Mohammad Mirjalili, Andrew Lewis,
Grey Wolf Optimizer,
Advances in Engineering Software,
Volume 69,
2014,
Pages 46-61,
https://doi.org/10.1016/j.advengsoft.2013.12.007.
"""


import numpy as np


E = 30000000
G = 12000000
L = 14
tau_max = 13600
sigma_max = 30000
delta_max = 0.25
P = 6000
C1 = 0.10471
C2 = 0.04811
C3 = 1

"""
Welded Beam Design
"""
def WBD_obj(X):
    """
    :param X:
    0.1 <= X[0] <= 2,
    0.1 <= X[1] <= 10,
    0.1 <= X[2] <= 10,
    0.1 <= X[3] <= 2
    :return:
    """
    V_weld = X[0] ** 2 * X[1]
    V_bar = X[2] * X[3] * (L + X[1])
    return (C1 + C3) * V_weld + C2 * V_bar


def WBD_cons(X):
    """
    :return: All cons should be minus than 0
    """
    con1 = tau_max - tau(X)
    con2 = delta_max - delta(X)
    con3 = X[3] - X[0]
    con4 = 5 - (1 + C1) * X[0] ** 2 + C2 * X[2] * X[3] * (L + X[1])
    con5 = X[0] - 0.125
    con6 = delta_max - delta(X)
    con7 = Pc(X) - P
    return [-con1, -con2, -con3, -con4, -con5, -con6, -con7]


def tau(X):
    return np.sqrt(tau_d(X) ** 2 + 2 * tau_d(X) * tau_dd(X) * X[1] / (2 * R(X)) + tau_dd(X) ** 2)


def tau_d(X):
    return P / (np.sqrt(2) * X[0] * X[1])


def tau_dd(X):
    return M(X) * R(X) / J(X)


def R(X):
    return np.sqrt(X[1] ** 2 / 4 + ((X[0] + X[2]) / 2) ** 2)


def M(X):
    return P * (L + X[1] / 2)


def J(X):
    return 2 * (X[0] * X[1] * np.sqrt(2) * (X[1] ** 2 / 12 + ((X[0] + X[2]) / 2) ** 2))


def sigma(X):
    return 6 * P * L / (X[3] * X[2] ** 2)


def delta(X):
    return 4 * P * L ** 3 / (E * X[3] * X[2] ** 2)


def Pc(X):
    coef = 4.013 * E * np.sqrt(X[2] ** 2 * X[3] ** 6 / 36) / (L ** 2)
    return coef * (1 - X[2] / (2 * L) * np.sqrt(E / (4 * G)))
