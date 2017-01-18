import numpy as np

from sklearn.model_selection import train_test_split
from neupy import algorithms, layers, environment
from neupy.datasets import reber


environment.reproducible()
environment.speedup()


def add_padding(data):
    n_sampels = len(data)
    max_seq_length = max(map(len, data))

    data_matrix = np.zeros((n_sampels, max_seq_length))
    for i, sample in enumerate(data):
        data_matrix[i, -len(sample):] = sample

    return data_matrix


data, labels = reber.make_reber_classification(n_samples=10000)

index_data = []
for sample in data:
    word = [reber.avaliable_letters.index(letter) + 1 for letter in sample]
    index_data.append(word)

index_data = add_padding(index_data)

x_train, x_test, y_train, y_test = train_test_split(
    index_data, labels, train_size=0.8)

n_categories = len(reber.avaliable_letters) + 1

network = algorithms.RMSProp(
    [
        layers.Input(1),
        layers.Embedding(n_categories, 10),
        layers.LSTM(20),
        layers.Sigmoid(1),
    ],

    step=0.05,
    verbose=True,
    batch_size=64,
    error='binary_crossentropy',
)
network.train(x_train, y_train, x_test, y_test, epochs=20)

y_predicted = network.predict(x_test).round()
y_predicted = y_predicted[:, 0]
accuracy = (y_predicted == y_test).mean()
print("Test accuracy: {:.2%}".format(accuracy))
