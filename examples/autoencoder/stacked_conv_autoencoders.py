from __future__ import division

import numpy as np
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from neupy import algorithms, layers, environment


environment.reproducible()

mnist = datasets.fetch_mldata('MNIST original')
data = mnist.data / 255.

target_scaler = OneHotEncoder()
target = mnist.target.reshape((-1, 1))
target = target_scaler.fit_transform(target).todense()

# Originaly we should have 70000 images from the MNIST dataset, but
# we will use only 1000 training example, All data that doesn't have
# labels we use to train features in the convolutional autoencoder.
n_labeled = 1000
n_samples = len(data)
n_unlabeled = n_samples - n_labeled

x_labeled, x_unlabeled, y_labeled, y_unlabeled = train_test_split(
    data.astype(np.float32),
    target.astype(np.float32),
    test_size=(1 - n_labeled / n_samples)
)

x_labeled_4d = x_labeled.reshape((n_labeled, 28, 28, 1))
x_unlabeled_4d = x_unlabeled.reshape((n_unlabeled, 28, 28, 1))

# We will features trained in the encoder and the first part for the future
# classifier. At first we pre-train them with unlabeled data, since we have
# a lot of it and we hope to learn some common features from it.
encoder = layers.join(
    layers.Input((28, 28, 1)),

    layers.Convolution((3, 3, 16)) > layers.Relu(),
    layers.Convolution((3, 3, 16)) > layers.Relu(),
    layers.MaxPooling((2, 2)),

    layers.Convolution((3, 3, 32)) > layers.Relu(),
    layers.MaxPooling((2, 2)),

    layers.Reshape(),

    layers.Relu(256),
    layers.Relu(128),
)

# Notice that in the decoder every operation reverts back changes from the
# encoder layer. Upscale replaces MaxPooling and Convolutional layer
# without padding replaced with large padding that increase size of the image.
decoder = layers.join(
    layers.Relu(256),
    layers.Relu(32 * 5 * 5),

    layers.Reshape((5, 5, 32)),

    layers.Upscale((2, 2)),
    layers.Convolution((3, 3, 16), padding=2) > layers.Relu(),

    layers.Upscale((2, 2)),
    layers.Convolution((3, 3, 16), padding=2) > layers.Relu(),
    layers.Convolution((3, 3, 1), padding=2) > layers.Sigmoid(),

    layers.Reshape(),
)

conv_autoencoder = algorithms.Momentum(
    connection=encoder > decoder,
    verbose=True,
    step=0.1,
    momentum=0.99,
    shuffle_data=True,
    batch_size=64,
    error='rmse',
)
conv_autoencoder.architecture()
conv_autoencoder.train(
    x_unlabeled_4d, x_unlabeled,
    x_labeled_4d, x_labeled,
    epochs=1,
)

# In order to speed up training for the upper layers we generate
# output from the encoder. In this way we won't need to regenerate
# encoded inputs for every epoch.
x_labeled_encoded = encoder.predict(x_labeled_4d)
x_unlabeled_encoded = encoder.predict(x_unlabeled_4d)

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
encoder_classifier.train(
    x_labeled_encoded, y_labeled,
    x_unlabeled_encoded, y_unlabeled,
    epochs=400,
)

# The final part of training is to put encoder and final classifier layers
# in order to fine tune network parameters before finilizing it's prediction
classifier = algorithms.GradientDescent(
    encoder > classifier_network,
    verbose=True,
    step=0.005,
    shuffle_data=True,
    batch_size=64,
    error='categorical_crossentropy',

    decay_rate=0.02,
    addons=[algorithms.WeightDecay],
)
classifier.architecture()
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
