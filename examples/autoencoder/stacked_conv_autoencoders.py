from __future__ import division

import theano
import numpy as np
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from neupy import algorithms, layers, environment


environment.reproducible()
theano.config.floatX = 'float32'
theano.config.allow_gc = False

mnist = datasets.fetch_mldata('MNIST original')
data = mnist.data / 255.

target_scaler = OneHotEncoder()
target = mnist.target.reshape((-1, 1))
target = target_scaler.fit_transform(target).todense()

n_labeled = 1000
n_samples = len(data)
n_unlabeled = n_samples - n_labeled

x_labeled, x_unlabeled, y_labeled, y_unlabeled = train_test_split(
    data.astype(np.float32),
    target.astype(np.float32),
    train_size=(n_labeled / n_samples)
)

x_labeled_4d = x_labeled.reshape((n_labeled, 1, 28, 28))
x_unlabeled_4d = x_unlabeled.reshape((n_unlabeled, 1, 28, 28))

encoder = layers.join(
    layers.Input((1, 28, 28)),

    layers.Convolution((16, 3, 3)) > layers.Relu(),
    layers.Convolution((16, 3, 3)) > layers.Relu(),
    layers.MaxPooling((2, 2)),

    layers.Convolution((32, 3, 3)) > layers.Relu(),
    layers.MaxPooling((2, 2)),

    layers.Reshape(),

    layers.Relu(256),
    layers.Relu(128),
)

decoder = layers.join(
    layers.Relu(256),
    layers.Relu(32 * 5 * 5),

    layers.Reshape((32, 5, 5)),

    layers.Upscale((2, 2)),
    layers.Convolution((16, 3, 3), padding='full') > layers.Relu(),

    layers.Upscale((2, 2)),
    layers.Convolution((16, 3, 3), padding='full') > layers.Relu(),
    layers.Convolution((1, 3, 3), padding='full') > layers.Sigmoid(),

    layers.Reshape(),
)

conv_autoencoder = algorithms.Momentum(
    connection=encoder > decoder,
    verbose=True,
    step=0.1,
    momentum=0.99,
    shuffle_data=True,
    batch_size=64,
    error='binary_crossentropy',
)
conv_autoencoder.architecture()
conv_autoencoder.train(x_unlabeled_4d, x_unlabeled,
                       x_labeled_4d, x_labeled, epochs=10)

x_labeled_encoded = encoder.output(x_labeled_4d).eval()
x_unlabeled_encoded = encoder.output(x_unlabeled_4d).eval()

classifier_network = layers.join(
    layers.PRelu(512),
    layers.Dropout(0.25),
    layers.Softmax(10),
)

encoder_classifier = algorithms.Adadelta(
    layers.Input(encoder.output_shape) > classifier_network,
    verbose=True,
    step=0.05,
    shuffle_data=True,
    batch_size=64,
    error='categorical_crossentropy',
)
encoder_classifier.architecture()
encoder_classifier.train(x_labeled_encoded, y_labeled,
                         x_unlabeled_encoded, y_unlabeled, epochs=100)

classifier = algorithms.MinibatchGradientDescent(
    encoder > classifier_network,
    verbose=True,
    step=0.01,
    shuffle_data=True,
    batch_size=64,
    error='categorical_crossentropy',
)
classifier.architecture()
classifier.train(x_labeled_4d, y_labeled, epochs=100)

unlabeled_predicted = classifier.predict(x_unlabeled_4d).argmax(axis=1)
y_unlabeled_classes = np.asarray(y_unlabeled).argmax(axis=1)

print(metrics.classification_report(y_unlabeled_classes, unlabeled_predicted))
score = metrics.accuracy_score(y_unlabeled_classes, unlabeled_predicted)
print("Validation accuracy: {:.2%}".format(score))
