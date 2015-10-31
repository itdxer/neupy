import numpy as np


xor_input_train = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
xor_target_train = np.array([[1, -1, -1, 1]]).T

simple_input_train = np.array([
    [0.1, 0.1, 0.2],
    [0.2, 0.3, 0.4],
    [0.1, 0.7, 0.2],
])
simple_target_train = np.array([
    [0.2, 0.2],
    [0.3, 0.3],
    [0.5, 0.5],
])
