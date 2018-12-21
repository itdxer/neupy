import numpy as np

from sklearn import datasets
from sklearn.model_selection import StratifiedShuffleSplit

from neupy.utils import asfloat


xor_x_train = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
xor_y_train = np.array([[1, -1, -1, 1]]).T

simple_x_train = asfloat(np.array([
    [0.1, 0.1, 0.2],
    [0.2, 0.3, 0.4],
    [0.1, 0.7, 0.2],
]))
simple_y_train = asfloat(np.array([
    [0.2, 0.2],
    [0.3, 0.3],
    [0.5, 0.5],
]))


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
    X, y = datasets.make_classification(
        n_samples=n_samples,
        n_features=n_features,
        random_state=random_state,
    )
    shuffle_split = StratifiedShuffleSplit(
        n_splits=1,
        train_size=0.6,
        test_size=0.1,
        random_state=random_state,
    )

    train_index, test_index = next(shuffle_split.split(X, y))
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    return x_train, x_test, y_train, y_test
