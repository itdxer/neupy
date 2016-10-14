import numpy as np

from sklearn import datasets
from sklearn.model_selection import StratifiedShuffleSplit


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

lenses = np.array([
    [1, 1, 1, 1, 1, 3],
    [2, 1, 1, 1, 2, 2],
    [3, 1, 1, 2, 1, 3],
    [4, 1, 1, 2, 2, 1],
    [5, 1, 2, 1, 1, 3],
    [6, 1, 2, 1, 2, 2],
    [7, 1, 2, 2, 1, 3],
    [8, 1, 2, 2, 2, 1],
    [9, 2, 1, 1, 1, 3],
    [10, 2, 1, 1, 2, 2],
    [11, 2, 1, 2, 1, 3],
    [12, 2, 1, 2, 2, 1],
    [13, 2, 2, 1, 1, 3],
    [14, 2, 2, 1, 2, 2],
    [15, 2, 2, 2, 1, 3],
    [16, 2, 2, 2, 2, 3],
    [17, 3, 1, 1, 1, 3],
    [18, 3, 1, 1, 2, 3],
    [19, 3, 1, 2, 1, 3],
    [20, 3, 1, 2, 2, 1],
    [21, 3, 2, 1, 1, 3],
    [22, 3, 2, 1, 2, 2],
    [23, 3, 2, 2, 1, 3],
    [24, 3, 2, 2, 2, 3],
])


def simple_classification(n_samples=100, n_features=10, random_state=33):
    """
    Generate simple classification task for training.

    Parameters
    ----------
    n_samples : int
        Number of samples in dataset.
    n_features : int
        Number of features for each sample.
    random_state : int
        Random state to make results reproducible.

    Returns
    -------
    tuple
        Returns tuple that contains 4 variables. There are input train,
        input test, target train, target test respectevly.
    """
    X, y = datasets.make_classification(n_samples=n_samples,
                                        n_features=n_features,
                                        random_state=random_state)
    shuffle_split = StratifiedShuffleSplit(n_splits=1, train_size=0.6,
                                           random_state=random_state)

    train_index, test_index = next(shuffle_split.split(X, y))
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    return x_train, x_test, y_train, y_test
