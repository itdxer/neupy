Convolutional layers
====================

NeuPy supports CNN architectures. Here is a simple example that solves MNIST problem.

.. code-block:: python

    from neupy import algorithms, layers
    from data import load_mnist

    x_train, y_train, x_test, y_test = load_mnist()

    network = algorithms.Adadelta(
        [
            layers.Convolution((32, 1, 3, 3)),
            layers.Relu(),
            layers.Convolution((48, 32, 3, 3)),
            layers.Relu(),
            layers.MaxPooling((2, 2)),
            layers.Dropout(0.2),

            layers.Reshape(),

            layers.Relu(200),
            layers.Dropout(0.3),
            layers.Softmax(10),
        ],

        error='categorical_crossentropy',
        step=1.0,
        verbose=True,
        shuffle_data=True,
    )
    network.train(x_train, y_train, x_test, y_test, epochs=6)
