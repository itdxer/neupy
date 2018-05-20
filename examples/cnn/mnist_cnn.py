import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import model_selection, metrics, datasets
from neupy import algorithms, layers, environment


environment.reproducible()
environment.speedup()


def load_data():
    mnist = datasets.fetch_mldata('MNIST original')
    data = mnist.data

    target_scaler = OneHotEncoder()
    target = mnist.target.reshape((-1, 1))
    target = target_scaler.fit_transform(target).todense()

    n_samples = data.shape[0]
    data = data.reshape((n_samples, 1, 28, 28))

    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        data.astype(np.float32),
        target.astype(np.float32),
        test_size=(1 / 7.)
    )

    mean = x_train.mean(axis=(0, 2, 3))
    std = x_train.std(axis=(0, 2, 3))

    x_train -= mean
    x_train /= std
    x_test -= mean
    x_test /= std

    return x_train, x_test, y_train, y_test


network = algorithms.Adadelta(
    [
        layers.Input((1, 28, 28)),

        layers.Convolution((32, 3, 3)) > layers.BatchNorm() > layers.Relu(),
        layers.Convolution((48, 3, 3)) > layers.BatchNorm() > layers.Relu(),
        layers.MaxPooling((2, 2)),

        layers.Convolution((64, 3, 3)) > layers.BatchNorm() > layers.Relu(),
        layers.MaxPooling((2, 2)),

        layers.Reshape(),
        layers.Linear(1024) > layers.BatchNorm() > layers.Relu(),
        layers.Softmax(10),
    ],

    # Using categorical cross-entropy as a loss function.
    # It's suitable for classification with 3 and more classes.
    error='categorical_crossentropy',

    # Min-batch size
    batch_size=128,

    # Learning rate. We can allow high values
    # since we are using Batch Normalization
    step=1.0,

    # Shows information about algorithm and
    # training progress in terminal
    verbose=True,

    # Randomly shuffles training dataset before every epoch
    shuffle_data=True,

    # Step decay algorithm minimizes learning step
    # monotonically after each iteration.
    addons=[algorithms.StepDecay],
    # Parameter controls step redution frequency. The higher
    # the value the slower step parameter decreases.
    reduction_freq=8,
)

# Shows networks architecture in terminal's output
network.architecture()

x_train, x_test, y_train, y_test = load_data()

# Train for only two epochs
network.train(x_train, y_train, x_test, y_test, epochs=2)

y_predicted = network.predict(x_test).argmax(axis=1)
y_test_labels = np.asarray(y_test.argmax(axis=1)).reshape(len(y_test))

print(metrics.classification_report(y_test_labels, y_predicted))
score = metrics.accuracy_score(y_test_labels, y_predicted)
print("Validation accuracy: {:.2%}".format(score))
