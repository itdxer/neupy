import matplotlib.pyplot as plt
from sklearn import datasets
from neupy import algorithms, utils
from neupy.utils import asfloat, tensorflow_session


def plot_rbm_components(rbm_network):
    session = tensorflow_session()
    weight = session.run(rbm_network.weight)

    plt.figure(figsize=(10, 10))
    plt.suptitle('RBM componenets', size=16)

    for index, image in enumerate(weight.T, start=1):
        plt.subplot(10, 10, index)
        plt.imshow(image.reshape((28, 28)), cmap=plt.cm.gray)

        plt.xticks([])
        plt.yticks([])

    plt.show()


utils.reproducible()

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
rbm.train(data, data, epochs=10)
plot_rbm_components(rbm)
