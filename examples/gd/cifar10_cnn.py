import theano
import numpy as np
from skdata.cifar10 import dataset, view
from sklearn import metrics, preprocessing
from neupy import algorithms, layers, environment


environment.reproducible()
theano.config.floatX = 'float32'

cifar10_dataset = dataset.CIFAR10()
cifar10_dataset.fetch(download_if_missing=True)

cifar10 = view.OfficialImageClassificationTask()
x_train, x_test = cifar10.train.x, cifar10.test.x
y_train, y_test = cifar10.train.y, cifar10.test.y

x_train = np.transpose(x_train, (0, 3, 1, 2))
x_test = np.transpose(x_test, (0, 3, 1, 2))
x_train, x_test = x_train / 255., x_test / 255.

target_scaler = preprocessing.OneHotEncoder()
y_train = target_scaler.fit_transform(y_train.reshape((-1, 1))).todense()
y_test = target_scaler.transform(y_test.reshape((-1, 1))).todense()

network = algorithms.Adadelta(
    [
        layers.Convolution((30, 3, 3, 3), border_mode='full'),
        layers.Relu(),
        layers.Convolution((30, 30, 3, 3)),
        layers.Relu(),
        layers.MaxPooling((2, 2)),
        layers.Dropout(proba=0.2),

        layers.Convolution((60, 30, 3, 3), border_mode='full'),
        layers.Relu(),
        layers.Convolution((60, 60, 3, 3)),
        layers.Relu(),
        layers.MaxPooling((2, 2)),
        layers.Dropout(proba=0.2),

        layers.Convolution((60, 60, 3, 3), border_mode='full'),
        layers.Relu(),
        layers.Convolution((60, 60, 3, 3)),
        layers.Relu(),
        layers.MaxPooling((2, 2)),
        layers.Dropout(proba=0.2),

        layers.Reshape(),

        layers.Relu(60 * 4 * 4),
        layers.Dropout(proba=0.5),
        layers.Softmax(600),
        layers.ArgmaxOutput(10),
    ],

    error='categorical_crossentropy',
    step=1,
    shuffle_data=True,
    verbose=True,

    epochs_step_minimizator=8,
    addons=[algorithms.SimpleStepMinimization]
)
network.architecture()
network.train(x_train, y_train, x_test, y_test, epochs=20)

y_predicted = network.predict(x_test)
y_test_labels = np.asarray(y_test.argmax(axis=1)).reshape(len(y_test))

print(metrics.classification_report(y_test_labels, y_predicted))
score = metrics.accuracy_score(y_test_labels, y_predicted)
print("Validation accuracy: {:.2f}%".format(100 * score))
