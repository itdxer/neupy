import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from neupy import algorithms, layers, environment


environment.reproducible()
environment.speedup()

mnist = datasets.fetch_mldata('MNIST original')

data = (mnist.data / 255.).astype(np.float32)

np.random.shuffle(data)
x_train, x_test = data[:60000], data[60000:]
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

        layers.Relu(128),
        layers.Relu(16),
        layers.Relu(128),
        layers.Relu(800),

        layers.Reshape((32, 5, 5)),

        layers.Upscale((2, 2)),
        layers.Convolution((16, 3, 3), padding='full') > layers.Relu(),

        layers.Upscale((2, 2)),
        layers.Convolution((16, 3, 3), padding='full') > layers.Relu(),
        layers.Convolution((1, 3, 3), padding='full') > layers.Sigmoid(),

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
conv_autoencoder.train(x_train_4d, x_train, x_test_4d, x_test, epochs=100)

n_samples = 4
images = x_test[:n_samples] * 255.
predicted_images = conv_autoencoder.predict(x_test_4d[:n_samples])
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
