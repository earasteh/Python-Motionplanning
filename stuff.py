"""
This file is just to test things and experiment outside the main project script.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from libs.cubic_spline_interpolator import generate_cubic_spline

# class Path:
#
#     def __init__(self):
#
#         # Get path to waypoints.csv
#         dir_path = 'data/waypoints2.csv'
#         df = pd.read_csv(dir_path)
#
#         x = df['X-axis'].values
#         y = df['Y-axis'].values
#         ds = 0.05
#
#         self.px, self.py, self.pyaw, _ = generate_cubic_spline(x, y, ds)


# path = Path()
# plt.figure(figsize = (8,6))
# plt.title('Path')
# plt.plot(path.px, path.py)
# plt.show()
#
# print(np.random.rand(2, 2))


# Fixing dugoff

g = 9.81
M = 3600 * 0.453592
L = 7 * 0.3048
ratio = 0.85
b = L / (1 + ratio)
a = L - b
I = 0.5 * M * a * b
C_f = 350 * 4.48 * 180 / np.pi
K_U = 1.1
C_r = K_U * (a / b) * C_f

N_f = M * g * b / L
N_r = M * g * a / L

s_f = 0
s_r = 0

mu_f = 0.85
mu_r = 0.85

alpha_f = np.linspace(-10, 10, 100)

# for alpha in alpha_f:
F_yfd = (C_f * np.tan(alpha_f)) / (1 - abs(s_f))
F_xfd = 0

lamda_f = mu_f / (2 * ((F_yfd / N_f) ** 2 + (F_xfd / N_f) ** 2) ** 0.5)

Y_f = np.zeros(100)

for i, lam in enumerate(lamda_f):
    if lam >= 1:
        Y_f[i] = F_yfd[i]
    else:
        Y_f[i] = F_yfd[i] * 2 * lam * (1 - lam / 2)


# plt.figure()
# plt.title('Front tire lateral force vs slip angle')
# # plt.plot(alpha_f1, Y_f1)
# plt.plot(alpha_f, Y_f)
# plt.show()

# xp = np.array([[9, -1, -0.2, 15.0], [12, -3, -4, 160], [13, 14, 17, 21]])