import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from neupy import algorithms, layers, environment


environment.reproducible()

mnist = datasets.fetch_mldata('MNIST original')
data = (mnist.data / 255.).astype(np.float32)

np.random.shuffle(data)
x_train, x_test = data[:60000], data[60000:]
x_train_4d = x_train.reshape((60000, 28, 28, 1))
x_test_4d = x_test.reshape((10000, 28, 28, 1))

conv_autoencoder = algorithms.Momentum(
    [
        layers.Input((28, 28, 1)),

        layers.Convolution((3, 3, 16)) > layers.Relu(),
        layers.Convolution((3, 3, 16)) > layers.Relu(),
        layers.MaxPooling((2, 2)),

        layers.Convolution((3, 3, 32)) > layers.Relu(),
        layers.MaxPooling((2, 2)),

        layers.Reshape(),

        layers.Relu(128),
        layers.Relu(16),

        # Notice that in the decoder every operation reverts back changes
        # from the encoder layer. Upscale replaces MaxPooling and
        # Convolutional layer without padding replaced with large padding
        # that increase size of the image.
        layers.Relu(128),

        # 800 is a shape that we got after we reshaped our image in the
        # Reshape layer
        layers.Relu(800),

        layers.Reshape((5, 5, 32)),

        # Upscaling layer reverts changes from the max pooling layer
        layers.Upscale((2, 2)),

        # If convolution operation in first layers with zero padding reduces
        # size of the image, then convolution with padding=2 increases size
        # of the image. It just does the opposite to the previous convolution
        layers.Convolution((3, 3, 16), padding=2) > layers.Relu(),

        layers.Upscale((2, 2)),
        layers.Convolution((3, 3, 16), padding=2) > layers.Relu(),
        layers.Convolution((3, 3, 1), padding=2) > layers.Sigmoid(),

        # We have to convert 4d tensor to the 2d in order to be
        # able to compute RMSE.
        layers.Reshape(),
    ],

    step=0.02,
    momentum=0.9,
    batch_size=128,
    error='rmse',

    shuffle_data=True,
    verbose=True,

    decay_rate=0.01,
    addons=[algorithms.WeightDecay],
)
conv_autoencoder.architecture()
conv_autoencoder.train(x_train_4d, x_train, x_test_4d, x_test, epochs=15)

n_samples = 6
images = x_test[:n_samples] * 255.
predicted_images = conv_autoencoder.predict(x_test_4d[:n_samples])
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
