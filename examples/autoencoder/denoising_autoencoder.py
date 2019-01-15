import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from neupy import algorithms, layers, utils


def load_data():
    X, _ = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
    X = (X / 255.).astype(np.float32)

    np.random.shuffle(X)
    x_train, x_test = X[:60000], X[60000:]

    return x_train, x_test


def visualize_reconstructions(autoencoder, x_test):
    n_samples = 4
    image_vectors = x_test[:n_samples, :]
    images = image_vectors * 255.
    predicted_images = autoencoder.predict(image_vectors)
    predicted_images = predicted_images * 255.

    # Compare real and reconstructed images
    fig, axes = plt.subplots(4, 2, figsize=(12, 8))
    iterator = zip(axes, images, predicted_images)

    for (left_ax, right_ax), real_image, predicted_image in iterator:
        real_image = real_image.reshape((28, 28))
        predicted_image = predicted_image.reshape((28, 28))

        left_ax.imshow(real_image, cmap=plt.cm.binary)
        right_ax.imshow(predicted_image, cmap=plt.cm.binary)

    plt.show()


if __name__ == '__main__':
    autoencoder = algorithms.Momentum(
        [
            layers.Input(784),
            layers.GaussianNoise(mean=0.5, std=0.1),
            layers.Sigmoid(100),
            layers.Sigmoid(784),
        ],
        step=0.1,
        verbose=True,
        momentum=0.9,
        nesterov=True,
        loss='rmse',
    )

    print("Preparing data...")
    x_train, x_test = load_data()

    print("Training autoencoder...")
    autoencoder.train(x_train, x_train, x_test, x_test, epochs=40)

    visualize_reconstructions(autoencoder, x_test)
