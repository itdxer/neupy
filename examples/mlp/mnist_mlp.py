import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import model_selection, metrics, datasets
from neupy import algorithms, layers


def load_data():
    X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
    X /= 255.
    X -= X.mean(axis=0)

    target_scaler = OneHotEncoder(sparse=False, categories='auto')
    y = target_scaler.fit_transform(y.reshape(-1, 1))

    return model_selection.train_test_split(
        X.astype(np.float32),
        y.astype(np.float32),
        test_size=(1 / 7.))


network = algorithms.Momentum(
    [
        layers.Input(784),
        layers.Relu(500),
        layers.Relu(300),
        layers.Softmax(10),
    ],

    # Using categorical cross-entropy as a loss function.
    # It's suitable for classification with 3 and more classes.
    loss='categorical_crossentropy',

    # Learning rate
    step=0.01,

    # Shows information about algorithm and
    # training progress in terminal
    verbose=True,

    # Randomly shuffles training dataset before every epoch
    shuffle_data=True,

    momentum=0.99,
    # Activates Nesterov momentum
    nesterov=True,
)

print("Preparing data...")
x_train, x_test, y_train, y_test = load_data()

print("Training...")
network.train(x_train, y_train, x_test, y_test, epochs=20)

y_predicted = network.predict(x_test).argmax(axis=1)
y_test = np.asarray(y_test.argmax(axis=1)).reshape(len(y_test))

print(metrics.classification_report(y_test, y_predicted))
score = metrics.accuracy_score(y_test, y_predicted)
print("Validation accuracy: {:.2%}".format(score))
