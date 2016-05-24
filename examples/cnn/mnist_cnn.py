import theano
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import cross_validation, metrics, datasets
from neupy import algorithms, layers, environment


environment.reproducible()
theano.config.floatX = 'float32'

mnist = datasets.fetch_mldata('MNIST original')
data = mnist.data

target_scaler = OneHotEncoder()
target = mnist.target.reshape((-1, 1))
target = target_scaler.fit_transform(target).todense()

n_samples = data.shape[0]
data = data.reshape((n_samples, 1, 28, 28))

x_train, x_test, y_train, y_test = cross_validation.train_test_split(
    data.astype(np.float32),
    target.astype(np.float32),
    train_size=(6 / 7.)
)

mean = x_train.mean(axis=(0, 2, 3))
std = x_train.std(axis=(0, 2, 3))

x_train -= mean
x_train /= std
x_test -= mean
x_test /= std

network = algorithms.Adadelta(
    [
        layers.Input((1, 28, 28)),

        layers.Convolution((32, 3, 3)) > layers.BatchNorm() > layers.Relu(),
        layers.Convolution((48, 3, 3)) > layers.BatchNorm() > layers.Relu(),
        layers.MaxPooling((2, 2)),

        layers.Convolution((64, 3, 3)) > layers.BatchNorm() > layers.Relu(),
        layers.MaxPooling((2, 2)),

        layers.Reshape(),
        layers.Linear(1024) > layers.BatchNorm() > layers.Relu(),
        layers.Softmax(10),
    ],

    error='categorical_crossentropy',
    step=1.0,
    verbose=True,
    shuffle_data=True,

    epochs_step_minimizator=8,
    addons=[algorithms.SimpleStepMinimization],
)
network.architecture()
network.train(x_train, y_train, x_test, y_test, epochs=15)

y_predicted = network.predict(x_test).argmax(axis=1)
y_test_labels = np.asarray(y_test.argmax(axis=1)).reshape(len(y_test))

print(metrics.classification_report(y_test_labels, y_predicted))
score = metrics.accuracy_score(y_test_labels, y_predicted)
print("Validation accuracy: {:.2f}%".format(100 * score))
