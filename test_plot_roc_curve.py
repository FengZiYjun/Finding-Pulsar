from plot_roc_curve import *
import numpy as np

y_test = np.array([1,1,0,1,0,1])
y_score = np.array([1,0,1,1,0,1])

plot_roc_curve(y_test, y_score)