import numpy as np


xor_input_train = np.array([[-1, -1],
                            [-1, 1],
                            [1, -1],
                            [1, 1]], dtype=np.float)
xor_target_train = np.array([[1],
                             [-1],
                             [-1],
                             [1]], dtype=np.float)


xor_zero_input_train = np.array([[0, 0],
                            [0, 1],
                            [1, 0],
                            [1, 1]], dtype=np.float)
xor_zero_target_train = np.array([[1],
                                  [0],
                                  [0],
                                  [1]], dtype=np.float)


even_input_train = np.array([[1, 2], [2, 1], [3, 1], [5, 1], [1, 6]])
even_target_train = np.array([[-1], [-1], [1], [1], [-1]])

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
