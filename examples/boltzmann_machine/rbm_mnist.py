import theano
import matplotlib.pyplot as plt
from sklearn import datasets
from neupy import algorithms, environment
from neupy.utils import asfloat


def plot_rbm_components(rbm_network):
    weight = rbm_network.weight.get_value()

    plt.figure(figsize=(10, 10))
    plt.suptitle('RBM componenets', size=16)

    for index, image in enumerate(weight.T, start=1):
        plt.subplot(10, 10, index)
        plt.imshow(image.reshape((28, 28)), cmap=plt.cm.gray)

        plt.xticks([])
        plt.yticks([])

    plt.show()


environment.reproducible()
theano.config.floatX = 'float32'

mnist = datasets.fetch_mldata('MNIST original')
data = asfloat(mnist.data > 130)

rbm = algorithms.RBM(
    n_visible=784,
    n_hidden=100,
    step=0.01,
    batch_size=20,

    verbose=True,
    shuffle_data=True,
)
rbm.train(data, epochs=10)

plot_rbm_components(rbm)
