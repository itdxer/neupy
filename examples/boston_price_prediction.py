import numpy as np
from sklearn import datasets, preprocessing
from sklearn.cross_validation import train_test_split
from neupy import algorithms, layers
from neupy.functions import rmsle


np.random.seed(0)

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
        layers.SigmoidLayer(13),
        layers.SigmoidLayer(50),
        layers.OutputLayer(1),
    ],
    search_method='golden',
    show_epoch=25,
    optimizations=[algorithms.LinearSearch],
)

cgnet.train(x_train, y_train, x_test, y_test, epochs=100)
cgnet.plot_errors()

y_predict = cgnet.predict(x_test).round(1)
error = rmsle(target_scaler.inverse_transform(y_test),
              target_scaler.inverse_transform(y_predict))
print("RMSLE = {}".format(error))
