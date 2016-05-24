import theano
import numpy as np
from sklearn import datasets, cross_validation
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from neupy.network.utils import shuffle
from neupy import algorithms, layers, environment, surgery


environment.reproducible()
theano.config.floatX = 'float32'
theano.config.allow_gc = False

mnist = datasets.fetch_mldata('MNIST original')
data = mnist.data / 255.

target_scaler = OneHotEncoder()
target = mnist.target.reshape((-1, 1))
target = target_scaler.fit_transform(target).todense()

x_train, x_test, y_train, y_test = cross_validation.train_test_split(
    data.astype(np.float32),
    target.astype(np.float32),
    train_size=(6 / 7.)
)

x_train_4d = x_train.reshape((60000, 1, 28, 28))
x_test_4d = x_test.reshape((10000, 1, 28, 28))

conv_autoencoder = algorithms.Momentum(
    [
        layers.Input((1, 28, 28)),

        layers.Convolution((16, 3, 3)) > layers.Relu(),
        layers.Convolution((16, 3, 3)) > layers.Relu(),
        layers.MaxPooling((2, 2)),

        layers.Convolution((32, 3, 3)) > layers.Relu(),
        layers.MaxPooling((2, 2)),

        layers.Reshape(),

        layers.Relu(512),
        layers.Relu(256),
        surgery.CutLine(),

        layers.Relu(512),
        layers.Relu(800),

        layers.Reshape((32, 5, 5)),

        layers.Upscale((2, 2)),
        layers.Convolution((16, 3, 3), border_mode='full') > layers.Relu(),

        layers.Upscale((2, 2)),
        layers.Convolution((16, 3, 3), border_mode='full') > layers.Relu(),
        layers.Convolution((1, 3, 3), border_mode='full') > layers.Sigmoid(),

        layers.Reshape(),
    ],

    verbose=True,
    step=0.1,
    momentum=0.99,
    shuffle_data=True,
    batch_size=128,
    error='rmse',
)
conv_autoencoder.architecture()
conv_autoencoder.train(x_train_4d, x_train, x_test_4d, x_test, epochs=10)

classifier_structure, _ = surgery.cut_along_lines(conv_autoencoder)

train_4d = classifier_structure.output(x_train_4d).eval()
test_4d = classifier_structure.output(x_test_4d).eval()

linear_classifier = algorithms.GradientDescent(
    [
        layers.Input(classifier_structure.output_shape),
        # layers.BatchNorm(),
        # layers.Relu(128),
        # layers.BatchNorm(),
        layers.Softmax(10),
    ],
    verbose=True,
    step=0.1,
    # momentum=0.99,
    shuffle_data=True,
    batch_size=128,
    error='categorical_crossentropy',
)
linear_classifier.architecture()
linear_classifier.train(train_4d, y_train, test_4d, y_test, epochs=100)

classification_layer = surgery.cut(linear_classifier, start=1, end=2)
classifier_structure = surgery.sew_together([classifier_structure,
                                             classification_layer])

classifier = algorithms.Adadelta(
    classifier_structure,
    verbose=True,
    step=0.1,
    # momentum=0.99,
    shuffle_data=True,
    batch_size=128,
    error='categorical_crossentropy',
)
classifier.architecture()
classifier.train(x_train_4d, y_train, x_test_4d, y_test, epochs=1)
