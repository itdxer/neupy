import numpy as np
from sklearn import datasets, preprocessing
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from neupy import algorithms, layers


np.random.seed(0)
plt.style.use('ggplot')

import theano
theano.config.mode = "FAST_COMPILE"
theano.config.optimizer = "fast_compile"
theano.config.allow_gc = False


def rmsle(actual, expected):
    count_of = expected.shape[0]
    square_logarithm_difference = np.log((actual + 1) / (expected + 1)) ** 2
    return np.sqrt((1 / count_of) * np.sum(square_logarithm_difference))


dataset = datasets.load_boston()
data, target = dataset.data, dataset.target

data_scaler = preprocessing.MinMaxScaler()
target_scaler = preprocessing.MinMaxScaler()

data = data_scaler.fit_transform(data)
target = target_scaler.fit_transform(target)

x_train, x_test, y_train, y_test = train_test_split(
    data, target, train_size=0.85
)

cgnet = algorithms.ConjugateGradient(
    connection=[
        layers.Sigmoid(13),
        layers.Sigmoid(50),
        layers.Sigmoid(10),
        layers.Output(1),
    ],
    show_epoch='100 times',
    verbose=True,
    optimizations=[
        algorithms.LinearSearch
    ],
)

cgnet.train(x_train, y_train, x_test, y_test, epochs=250)
y_predict = cgnet.predict(x_test)

error = rmsle(target_scaler.inverse_transform(y_test),
              target_scaler.inverse_transform(y_predict).T.round(1))
print("RMSLE = {}".format(error))

from neupy.network import errors
error = errors.mse(target_scaler.inverse_transform(y_test),
              target_scaler.inverse_transform(y_predict).T).eval()
print("MSE = {}".format(error))
