import numpy as np
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold

from neupy.algorithms import PNN


dataset = datasets.load_iris()
data, target = dataset.data, dataset.target

print("> Start classify iris dataset")
skfold = StratifiedKFold(n_splits=10)

for i, (train, test) in enumerate(skfold.split(data, target), start=1):
    x_train, x_test = data[train], data[test]
    y_train, y_test = target[train], target[test]

    pnn_network = PNN(std=0.1, verbose=False)
    pnn_network.train(x_train, y_train)
    result = pnn_network.predict(x_test)

    n_predicted_correctly = np.sum(result == y_test)
    n_test_samples = test.size

    print("Test #{:<2}: Guessed {} out of {}".format(
        i, n_predicted_correctly, n_test_samples))
