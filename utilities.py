import numpy as np

def calculate_l(s):
    l = np.array([1])
    for s_x in s:
        l = np.append(l, l[-1] * s_x)

    return l

def calculate_s(l):
    s = np.array([])
    for i in range(l.size - 1):
        s = np.append(s, l[i+1] / l[i])

    return s