import itertools

import theano
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from skimage.filters import threshold_adaptive
from neupy import algorithms, environment
from neupy.utils import asfloat


def show_image(ax, image):
    image_shape = (62, 47)
    ax.imshow(image.reshape(image_shape), cmap=plt.cm.gray)


def disable_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def plot_rbm_sampled_images(rbm_network, data, training_data):
    n_samples = len(training_data)
    iterations = [0, 0, 1, 10, 100, 1000]

    nrows, ncols = (6, 6)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 16))

    titles = [
        'Raw image',
        'Binarized image',
        'Sample after\n1 iteration',
        'Sample after\n10 iterations',
        'Sample after\n100 iterations',
        'Sample after\n1000 iterations',
    ]

    for i, title in enumerate(titles):
        axes[0][i].set_title(title)

    for i, row_axes in enumerate(axes):
        image_index = np.random.randint(0, n_samples)
        binary_image = training_data[image_index]

        show_image(row_axes[0], data[image_index])
        show_image(row_axes[1], binary_image)

        for j in range(2, ncols):
            n_iter = iterations[j]
            sampled_image = rbm_network.gibbs_sampling(binary_image, n_iter)
            show_image(row_axes[j], sampled_image)

    for ax in itertools.chain(*axes):
        disable_ticks(ax)


def binarize_images(data):
    binarized_data = []
    for image in data:
        binary_adaptive = threshold_adaptive(image.reshape((62, 47)),
                                             block_size=15)
        binarized_data.append(binary_adaptive.ravel())
    return asfloat(binarized_data)


environment.reproducible()
theano.config.floatX = 'float32'

people_dataset = datasets.fetch_lfw_people()
data = people_dataset.data
np.random.shuffle(data)

binarized_data = binarize_images(data)

x_train, x_test, binarized_x_train, binarized_x_test = train_test_split(
    data, binarized_data, train_size=0.9
)

rbm = algorithms.RBM(
    n_visible=2914,
    n_hidden=1200,
    step=0.01,
    batch_size=10,

    verbose=True,
    shuffle_data=True,
)
rbm.train(binarized_x_train, binarized_x_test, epochs=70)

plot_rbm_sampled_images(rbm, x_test, binarized_x_test)
plt.show()
