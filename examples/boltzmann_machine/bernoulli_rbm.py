import theano
import matplotlib.pyplot as plt
from sklearn import datasets
from neupy import algorithms
from neupy.utils import asfloat


theano.config.floatX = 'float32'

mnist = datasets.fetch_mldata('MNIST original')
data = asfloat(mnist.data / 255.)

rbm = algorithms.RBM(
    n_visible=784,
    n_hidden=100,
    step=0.01,
    verbose=True,
)
rbm.train(data, epochs=10)

weight = rbm.weight.get_value()
plt.figure(figsize=(12, 12))

for i, row in enumerate(weight.T[:100, :], start=1):
    plt.subplot(10, 10, i)
    plt.imshow(row.reshape((28, 28)), cmap=plt.cm.binary)
    plt.xticks([], [])
    plt.yticks([], [])

plt.show()
