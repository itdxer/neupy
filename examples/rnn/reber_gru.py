import numpy as np

from sklearn.model_selection import train_test_split
from neupy import algorithms, layers
from neupy.datasets import reber


def add_padding(data):
    n_sampels = len(data)
    max_seq_length = max(map(len, data))

    data_matrix = np.zeros((n_sampels, max_seq_length))
    for i, sample in enumerate(data):
        data_matrix[i, -len(sample):] = sample

    return data_matrix


# An example of possible values for the `data` and `labels`
# variables
#
# >>> data
# array([array([1, 3, 1, 4]),
#        array([0, 3, 0, 3, 0, 4, 3, 0, 4, 4]),
#        array([0, 3, 0, 0, 3, 0, 4, 2, 4, 1, 0, 4, 0])], dtype=object)
# >>>
# >>> labels
# array([1, 0, 0])
data, labels = reber.make_reber_classification(
    n_samples=10000, return_indices=True)

# Shift all indices by 1. In the next row we will add zero
# paddings, so we need to make sure that we will not confuse
# paddings with zero indices.
data = data + 1

# Add paddings at the beggining of each vector to make sure
# that all samples has the same length. This trick allows to
# train network with multiple independent samples.
data = add_padding(data)

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2)

n_categories = len(reber.avaliable_letters) + 1  # +1 for zero paddings
n_time_steps = x_train.shape[1]

optimizer = algorithms.RMSProp(
    [
        layers.Input(n_time_steps),
        # shape: (n_samples, n_time_steps)

        layers.Embedding(n_categories, 10),
        # shape: (n_samples, n_time_steps, 10)

        # unroll_scan - speed up calculation for short sequences
        layers.GRU(20, unroll_scan=True),
        # shape: (n_samples, 20)

        layers.Sigmoid(1),
        # shape: (n_samples, 1)
    ],
    step=0.01,
    verbose=True,
    batch_size=64,
    loss='binary_crossentropy',
)
optimizer.train(x_train, y_train, x_test, y_test, epochs=20)

y_predicted = optimizer.predict(x_test).round()
accuracy = (y_predicted.T == y_test).mean()
print("Test accuracy: {:.2%}".format(accuracy))
