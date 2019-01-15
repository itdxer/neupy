from __future__ import division

import numpy as np
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from neupy.layers import *
from neupy import algorithms, layers


X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255.

target_scaler = OneHotEncoder(categories='auto', sparse=False)
y = target_scaler.fit_transform(y.reshape(-1, 1))

# Originaly we should have 70000 images from the MNIST dataset, but
# we will use only 1000 training example, All data that doesn't have
# labels we use to train features in the convolutional autoencoder.
n_labeled = 1000
n_samples = len(X)
n_unlabeled = n_samples - n_labeled

x_labeled, x_unlabeled, y_labeled, y_unlabeled = train_test_split(
    X.astype(np.float32),
    y.astype(np.float32),
    test_size=(1 - n_labeled / n_samples)
)

x_labeled_4d = x_labeled.reshape((n_labeled, 28, 28, 1))
x_unlabeled_4d = x_unlabeled.reshape((n_unlabeled, 28, 28, 1))

# We will features trained in the encoder and the first part for the future
# classifier. At first we pre-train them with unlabeled data, since we have
# a lot of it and we hope to learn some common features from it.
encoder = join(
    Input((28, 28, 1)),

    Convolution((3, 3, 16)) >> Relu(),
    Convolution((3, 3, 16)) >> Relu(),
    MaxPooling((2, 2)),

    Convolution((3, 3, 32)) >> Relu(),
    MaxPooling((2, 2)),

    Reshape(),

    Relu(256),
    Relu(128),
)

# Notice that in the decoder every operation reverts
# back changes from the encoder layer.
decoder = join(
    Relu(256),
    Relu(32 * 5 * 5),

    Reshape((5, 5, 32)),

    Upscale((2, 2)),
    Deconvolution((3, 3, 16)) >> Relu(),

    Upscale((2, 2)),
    Deconvolution((3, 3, 16)) >> Relu(),
    Deconvolution((3, 3, 1)) >> Sigmoid(),
)

conv_autoencoder = algorithms.Momentum(
    network=(encoder >> decoder),

    loss='rmse',
    step=0.02,
    batch_size=128,
    regularizer=algorithms.l2(0.001),

    shuffle_data=True,
    verbose=True,
)
conv_autoencoder.train(
    x_unlabeled_4d, x_unlabeled_4d,
    x_labeled_4d, x_labeled_4d,
    epochs=1,
)

# In order to speed up training for the upper layers we generate
# output from the encoder. In this way we won't need to regenerate
# encoded inputs for every epoch.
x_labeled_encoded = encoder.predict(x_labeled_4d)
x_unlabeled_encoded = encoder.predict(x_unlabeled_4d)

classifier_network = join(
    PRelu(512),
    Dropout(0.25),
    Softmax(10),
)

encoder_classifier = algorithms.Adadelta(
    Input(encoder.output_shape[1:]) >> classifier_network,
    verbose=True,
    step=0.05,
    shuffle_data=True,
    batch_size=64,
    loss='categorical_crossentropy',
)
encoder_classifier.train(
    x_labeled_encoded, y_labeled,
    x_unlabeled_encoded, y_unlabeled,
    epochs=400,
)

# The final part of training is to put encoder and final classifier layers
# in order to fine tune network parameters before finilizing it's prediction
classifier = algorithms.GradientDescent(
    network=(encoder >> classifier_network),
    verbose=True,
    step=0.005,
    shuffle_data=True,
    batch_size=64,
    loss='categorical_crossentropy',
    regularizer=algorithms.l2(0.02),
)
classifier.train(x_labeled_4d, y_labeled, epochs=100)
classifier.train(
    x_labeled_4d, y_labeled,
    x_unlabeled_4d, y_unlabeled,
    epochs=1,
)

unlabeled_predicted = classifier.predict(x_unlabeled_4d).argmax(axis=1)
y_unlabeled_classes = np.asarray(y_unlabeled).argmax(axis=1)

print(metrics.classification_report(y_unlabeled_classes, unlabeled_predicted))
score = metrics.accuracy_score(y_unlabeled_classes, unlabeled_predicted)
print("Validation accuracy: {:.2%}".format(score))
