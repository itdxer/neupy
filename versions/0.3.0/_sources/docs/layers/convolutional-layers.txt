Convolutional layers
====================

NeuPy supports CNN architectures. Here is a simple a network that aplies on the MNIST dataset.

.. code-block:: python

    from neupy import algorithms, layers
    from data import load_mnist

    x_train, y_train, x_test, y_test = load_mnist()

    network = algorithms.Adadelta(
        [
            layers.Input((1, 28, 28))

            layers.Convolution((32, 3, 3)),
            layers.Relu(),
            layers.Convolution((48, 3, 3)),
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

Convolution
***********

NeuPy supports only 2D convolution. It's trivial to make a 1D convoltion. You just need to set up width or height equal to 1. In this section I will focuse only on 2D convoltuin, but with a small modifications everything work for 1D as well.

In the previous example you should see a network that contains a couple of convolutional layers. Each of these layers takes one mandatory argument that defines convolutional layer structure. Each parameter excepts a tuple that contains three integers ``(output channels, filter rows, filter columns)``. Information about the input channel takes from the previous layer.

Convolutional layer has a few other attributes that you can modify. You can check the ':layer:`Convolution`' layer's documentation and find more information ralated to this layer type.

Pooling
*******

Pooling works very similar. As in the convolutional layer you also need to set up one mandatory attribute as a tuple. But in case of pooling layer this attribute should be a tuple that contains only two integers. This parameters defines a factor by which to downscale ``(vertical, horizontal)``. (2, 2) will halve the image in each dimension.

Pooling defined as for the 2D layers, but you also can use in case of 1D convolution. In that case you need to define one of the downscale factors equal to 1. For instance, it can be somethig like that ``(1, 2)``.
