import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

from neupy.layers import *
from neupy import algorithms


def load_data():
    X, _ = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
    X = (X / 255.).astype(np.float32)

    np.random.shuffle(X)
    x_train_2d, x_test_2d = X[:60000], X[60000:]
    x_train_4d = x_train_2d.reshape((60000, 28, 28, 1))
    x_test_4d = x_test_2d.reshape((10000, 28, 28, 1))

    return x_train_4d, x_test_4d


def visualize_reconstructions(x_test_4d, n_samples=6):
    x_test = x_test_4d.reshape(x_test_4d.shape[0], -1)
    images = x_test[:n_samples] * 255.
    predicted_images = optimizer.predict(x_test_4d[:n_samples])
    predicted_images = predicted_images * 255.

    # Compare real and reconstructed images
    fig, axes = plt.subplots(n_samples, 2, figsize=(12, 8))
    iterator = zip(axes, images, predicted_images)

    for (left_ax, right_ax), real_image, predicted_image in iterator:
        real_image = real_image.reshape((28, 28))
        predicted_image = predicted_image.reshape((28, 28))

        left_ax.imshow(real_image, cmap=plt.cm.binary)
        right_ax.imshow(predicted_image, cmap=plt.cm.binary)

    plt.show()


if __name__ == '__main__':
    network = join(
        Input((28, 28, 1)),

        Convolution((3, 3, 16)) >> Relu(),
        Convolution((3, 3, 16)) >> Relu(),
        MaxPooling((2, 2)),

        Convolution((3, 3, 32)) >> Relu(),
        MaxPooling((2, 2)),

        Reshape(),

        Relu(128),
        Relu(16),

        # Notice that in the decoder every operation reverts back
        # changes from the encoder layer.
        Relu(128),

        # 800 is a shape that we got after we reshaped our image in the
        # Reshape layer
        Relu(800),

        Reshape((5, 5, 32)),

        # Upscaling layer reverts changes from the max pooling layer
        Upscale((2, 2)),

        # Deconvolution (a.k.a Transposed Convolution) reverts
        # changes done by Convolution
        Deconvolution((3, 3, 16)) >> Relu(),

        Upscale((2, 2)),
        Deconvolution((3, 3, 16)) >> Relu(),
        Deconvolution((3, 3, 1)) >> Sigmoid()
    )
    optimizer = algorithms.Momentum(
        network,
        step=0.02,
        momentum=0.9,
        batch_size=128,
        loss='rmse',

        shuffle_data=True,
        verbose=True,

        regularizer=algorithms.l2(0.01),
    )

    x_train_4d, x_test_4d = load_data()
    optimizer.train(x_train_4d, x_train_4d, x_test_4d, x_test_4d, epochs=1)
    visualize_reconstructions(x_test_4d, n_samples=6)
