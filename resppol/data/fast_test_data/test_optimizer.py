#!/usr/bin/python
# version 2018/07/03

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import fmin_tnc
from scipy.optimize import minimize

x = np.array([2, 5, 7])
y = np.array([1, 3, 4])
kd = np.array([0, 0])


def linfit(kd):
    k = kd[0]
    d = kd[1]
    sse = 0.0
    for i in range(len(x)):
        sse += np.power((k * x[i] + d - y[i]), 2)

    return (sse)


minout = minimize(linfit, kd)
print(minout)
