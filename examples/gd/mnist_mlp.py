import theano
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import cross_validation, metrics, datasets
from neupy import algorithms, layers


np.random.seed(0)
theano.config.floatX = 'float32'

mnist = datasets.fetch_mldata('MNIST original')

target_scaler = OneHotEncoder()
target = mnist.target.reshape((-1, 1))
target = target_scaler.fit_transform(target).todense()

data = mnist.data / 255.
data = data - data.mean(axis=0)

x_train, x_test, y_train, y_test = cross_validation.train_test_split(
    data.astype(np.float32),
    target.astype(np.float32),
    train_size=(6 / 7.)
)

network = algorithms.Momentum(
    [
        layers.Relu(784),
        layers.Dropout(0.2),
        layers.Relu(400),
        layers.Dropout(0.2),
        layers.Softmax(400),
        layers.ArgmaxOutput(10),
    ],
    error='categorical_crossentropy',
    step=0.01,
    verbose=True,
    shuffle_data=True,
    momentum=0.99,
    nesterov=True,
    epochs_step_minimizator=10,
    addons=[algorithms.SimpleStepMinimization],
)
network.train(x_train, y_train, x_test, y_test, epochs=20)

y_predicted = network.predict(x_test)
y_test = np.asarray(y_test.argmax(axis=1)).reshape(len(y_test))

print(metrics.classification_report(y_test, y_predicted))
score = metrics.accuracy_score(y_test, y_predicted)
print("Validation accuracy: {:.2f}%".format(100 * score))
