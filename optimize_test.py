__author__ = 'Mat'

from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize

#-------------------------------

def f(c):
    return c[0] ** 2+2 + 2* c[0]

result = optimize.minimize(f, [3])
print result
