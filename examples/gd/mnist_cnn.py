import theano
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import cross_validation, metrics, datasets
from neupy import algorithms, layers, environment


environment.reproducible()
theano.config.floatX = 'float32'

mnist = datasets.fetch_mldata('MNIST original')

target_scaler = OneHotEncoder()
target = mnist.target.reshape((-1, 1))
target = target_scaler.fit_transform(target).todense()

data = mnist.data / 255.
data = data - data.mean(axis=0)

n_samples = data.shape[0]
data = data.reshape((n_samples, 1, 28, 28))

x_train, x_test, y_train, y_test = cross_validation.train_test_split(
    data.astype(np.float32),
    target.astype(np.float32),
    train_size=(6 / 7.)
)

network = algorithms.Adadelta(
    [
        layers.Convolution((32, 1, 3, 3)),
        layers.Relu(),
        layers.Convolution((48, 32, 3, 3)),
        layers.Relu(),
        layers.MaxPooling((2, 2)),
        layers.Dropout(0.2),

        layers.Reshape(),

        layers.Relu(6912),
        layers.Dropout(0.3),
        layers.Softmax(200),
        layers.ArgmaxOutput(10),
    ],

    error='categorical_crossentropy',
    step=1.0,
    verbose=True,
    shuffle_data=True,

    epochs_step_minimizator=8,
    addons=[algorithms.SimpleStepMinimization],
)
network.architecture()
network.train(x_train, y_train, x_test, y_test, epochs=6)

y_predicted = network.predict(x_test)
y_test_labels = np.asarray(y_test.argmax(axis=1)).reshape(len(y_test))

print(metrics.classification_report(y_test_labels, y_predicted))
score = metrics.accuracy_score(y_test_labels, y_predicted)
print("Validation accuracy: {:.2f}%".format(100 * score))
