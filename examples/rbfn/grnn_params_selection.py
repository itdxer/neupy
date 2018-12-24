import heapq

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from neupy import algorithms


def rmsle(expected, predicted):
    log_expected = np.log1p(expected + 1)
    log_predicted = np.log1p(predicted + 1)
    squared_log_error = np.square(log_expected - log_predicted)
    return np.sqrt(np.mean(squared_log_error))


def scorer(network, X, y):
    result = network.predict(X)
    return rmsle(result, y)


def report(results, n_top=3):
    ranks = heapq.nlargest(n_top, results['rank_test_score'])

    for i in ranks:
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


print("Run Random Search CV")
dataset = datasets.load_diabetes()
random_search = RandomizedSearchCV(
    algorithms.GRNN(std=0.1, verbose=False),
    param_distributions={'std': np.arange(1e-2, 1, 1e-3)},
    n_iter=100,
    cv=3,
    scoring=scorer,
)
random_search.fit(dataset.data, dataset.target)
report(random_search.cv_results_)
