import numpy as np
from sklearn import datasets
from sklearn.cross_validation import StratifiedKFold

from neuralpy.algorithms import PNN


dataset = datasets.load_iris()
data = dataset.data
target = dataset.target

test_data_size = 10
skfold = StratifiedKFold(target, test_data_size)
avarage_result = 0

print("> Start classify iris dataset")

for i, (train, test) in enumerate(skfold, start=1):
    x_train, x_test = data[train], data[test]
    y_train, y_test = target[train], target[test]

    pnn_network = PNN(standard_deviation=0.1, verbose=False)
    pnn_network.train(x_train, y_train)
    result = pnn_network.predict(x_test)
    print("Test #{:<2}: {}/{}".format(
        i, np.sum(result == y_test), test.size
    ))
