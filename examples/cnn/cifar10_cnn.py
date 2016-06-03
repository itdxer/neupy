import theano
import numpy as np
from skdata.cifar10 import dataset, view
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from neupy import algorithms, layers, environment


environment.reproducible()
theano.config.floatX = 'float32'
theano.config.allow_gc = False

cifar10_dataset = dataset.CIFAR10()
cifar10_dataset.fetch(download_if_missing=True)

cifar10 = view.OfficialImageClassificationTask()
x_train, x_test = cifar10.train.x, cifar10.test.x
y_train, y_test = cifar10.train.y, cifar10.test.y

x_train = np.transpose(x_train, (0, 3, 1, 2))
x_train = x_train.astype(np.float32, copy=False)

x_test = np.transpose(x_test, (0, 3, 1, 2))
x_test = x_test.astype(np.float32, copy=False)

mean = x_train.mean(axis=(0, 2, 3)).reshape((1, -1, 1, 1))
std = x_train.std(axis=(0, 2, 3)).reshape((1, -1, 1, 1))

x_train -= mean
x_train /= std
x_test -= mean
x_test /= std

target_scaler = OneHotEncoder()
y_train = target_scaler.fit_transform(y_train.reshape((-1, 1))).todense()
y_test = target_scaler.transform(y_test.reshape((-1, 1))).todense()

network = algorithms.Adadelta(
    [
        layers.Input((3, 32, 32)),

        layers.Convolution((64, 3, 3)) > layers.BatchNorm() > layers.PRelu(),
        layers.Convolution((64, 3, 3)) > layers.BatchNorm() > layers.PRelu(),
        layers.MaxPooling((2, 2)),

        layers.Convolution((128, 3, 3)) > layers.BatchNorm() > layers.PRelu(),
        layers.Convolution((128, 3, 3)) > layers.BatchNorm() > layers.PRelu(),
        layers.MaxPooling((2, 2)),

        layers.Reshape(),

        layers.Linear(1024) > layers.BatchNorm() > layers.PRelu(),
        layers.Linear(1024) > layers.BatchNorm() > layers.PRelu(),
        layers.Softmax(10),
    ],

    error='categorical_crossentropy',
    step=0.25,
    shuffle_data=True,
    batch_size=128,
    verbose=True,
)
network.architecture()
network.train(x_train, y_train, x_test, y_test, epochs=20)

y_predicted = network.predict(x_test).argmax(axis=1)
y_test_labels = np.asarray(y_test.argmax(axis=1)).reshape(len(y_test))

print(metrics.classification_report(y_test_labels, y_predicted))
score = metrics.accuracy_score(y_test_labels, y_predicted)
print("Validation accuracy: {:.2%}".format(score))
print(metrics.confusion_matrix(y_predicted, y_test_labels))
