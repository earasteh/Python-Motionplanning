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
