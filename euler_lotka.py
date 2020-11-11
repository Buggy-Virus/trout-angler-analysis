import numpy as np
from scipy.optimize import brentq

# The Euler-Lotka equation
def euler_lotka(x, l, m, r):
    return np.sum(l * m * np.exp(-r * x)) - 1

def parameterize_el(x, l, m):

    def el_parameterized(r):
        return np.sum(l * m * np.exp(-r * x)) - 1

    return el_parameterized

# Bracket the root and solve with scipy.optimize.brentq
def find_euler_lotka_r(x, l, m, a, w):
    r = brentq(parameterize_el(x, l, m), a, w)
    return r

# Calculate growth rate
def growth_rate(x, r):
    return np.exp(-r * x)

def discrete_growth(x, r):
    vfunc = np.vectorize(growth_rate)
    discrete_growth = vfunc(x, r)
    return np.stack(x, discrete_growth)