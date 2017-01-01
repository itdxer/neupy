import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from neupy import layers, algorithms, environment


environment.reproducible()
environment.speedup()


def make_dataset():
    data, target = make_classification(n_samples=10000, n_features=20)

    n_categorical = 3
    bins = np.array([0, 0.2, 0.5, 0.8, 1])

    # Create categorical features
    for i in range(n_categorical):
        data[:, i] = np.digitize(data[:, i], bins=bins)

    return train_test_split(data, target, train_size=0.9)


def only_numerical(data):
    return data[:, 3:]


class ConvertCategorical(object):
    def fit_transform(self, data):
        n_categories = np.max(data, axis=0) + 1
        self.index_shifts = np.cumsum(n_categories) - n_categories[0]
        return self.transform(data)

    def transform(self, data):
        return data + self.index_shifts


convert_categorical = ConvertCategorical()
x_train, x_test, y_train, y_test = make_dataset()

x_train_cat = convert_categorical.fit_transform(x_train[:, :3])
x_train_num = only_numerical(x_train)

# We use .max(), because each category has a unique identifier.
# Maximum value defines last category ID. Since first category has
# 0 identifier, we need to +1 to obtain total result
n_unique_categories = int(x_train_cat.max() + 1)

x_test_cat = convert_categorical.transform(x_test[:, :3])
x_test_num = only_numerical(x_test)

network = algorithms.Momentum(
    [
        [[
            # 3 categorical inputs
            layers.Input(3),

            # Train embedding matrix for categorical inputs.
            # It has 18 different unique categories (6 categories
            # per each of the 3 columns). Next layer projects each
            # category into 4 dimensional space. Output shape from
            # the layer should be: (batch_size, 3, 4)
            layers.Embedding(n_unique_categories, 4),

            # Reshape (batch_size, 3, 4) to (batch_size, 12)
            layers.Reshape(),
        ], [
            # 17 numerical inputs
            layers.Input(17),
        ]],

        # Concatenate (batch_size, 12) and (batch_size, 17)
        # into one matrix with shape (batch_size, 29)
        layers.Concatenate(),

        layers.Relu(128),  # > layers.Dropout(0.2),
        layers.Relu(32) > layers.Dropout(0.5),

        layers.Sigmoid(1)
    ],

    step=0.2,
    verbose=True,
    momentum=0.9,
    nesterov=True,
    error='binary_crossentropy',
)

network.train([x_train_num, x_train_cat], y_train,
              [x_test_num, x_test_cat], y_test,
              epochs=300)
y_predicted = network.predict([x_test_num, x_test_cat])

accuracy = accuracy_score(y_test, y_predicted.round())
print("Accuracy: {:.2%}".format(accuracy))
