Mixing NeuPy with Tensorflow
============================

NeuPy allows to get quickly from idea to the first prototype of the model, but in some cases, API can be too restrictive. Working directly with Tensorflow allows us to be very flexible with the code, even though you will need to write more code. In order to be able to use simple and convenient API provided by NeuPy and take advantage of the Tensorflow's flexibility, NeuPy provides with a direct access to all the inputs and outputs of the neural network models.

Let's start with an example, let's say we have simple CNN architecture that expects 28x28 grey images and it returns multinomial probability distribution across 10 possible output classes.

.. code-block:: python

    from neupy.layers import *

    Conv = Convolution.define(padding='same')
    network = join(
        Input((28, 28, 1)),

        Conv((3, 3, 18)) >> Relu(),
        MaxPooling((2, 2)),

        Conv((3, 3, 36)) >> Relu(),
        Conv((3, 3, 36)) >> Relu(),
        MaxPooling((2, 2)),

        Reshape(),

        Relu(256) >> Dropout(0.5),
        Sigmoid(10),
    )

Network's inputs and outputs
----------------------------

Nothing has happened at this point. We've defined architecture, but nothing has been added to the Tensorflow's graph. We can do it in two different ways. First, we can use ``outputs`` attribute and get access to the output tensor.

.. code-block:: python

    >>> network.outputs
    <tf.Tensor 'Sigmoid/Sigmoid:0' shape=(?, 10) dtype=float32>

When network ``outputs`` was triggered for the first time NeuPy creates placeholder that expects batch of the 28x28 images with single channel. After that, created placeholder will be propagated through the network and tensor, associated with the output from the final layer, will be returned. NeuPy caches output and each time ``outputs`` attribute triggered the same tensor will be returned.

.. code-block:: python

    >>> id(network.outputs)
    4851785344
    >>> id(network.outputs)
    4851785344

Placeholder has been created implicitly, but it's possible to get access to it by triggering the ``inputs`` method.

.. code-block:: python

    >>> network.inputs
    <tf.Tensor 'placeholder/input/input-1:0' shape=(?, 28, 28, 1) dtype=float32>

As for the ``outputs`` attribute, placeholder will be created in the lazy way and it will be cached and the same object will be returned each time we trigger ``inputs`` attribute.

Also, It's important to note that output from the ``inputs`` and ``outputs`` attributes will be a list, for cases, when network's architecture has multiple inputs or outputs.

In certain cases, we might want to propagate custom inputs through the network. It's possible to do it using the ``output`` method.

.. code-block:: python

    >>> import numpy as np
    >>> images = np.random.random((7, 28, 28, 1))
    >>>
    >>> output_tensor = network.output(images)
    <tf.Tensor 'Sigmoid_1/Sigmoid:0' shape=(7, 10) dtype=float32>

Basically, ``outputs`` attribute is just a shortcut for the ``network.output(network.inputs)``. The only difference is that output won't be cached when the same input is propagated multiple times through the network.

.. code-block:: python

    >>> id(network.output(images))
    4852735056
    >>> id(network.output(images))
    4853088496

Propagate inputs for training
-----------------------------

Certain layers might have different behavior during training and inference time. For example, we want to enable Dropout layer during the training and disable it during the inference time. NeuPy allows to pass different messages over the network with the input. For example, we can control training outputs with the ``training`` argument.

.. code-block:: python

    >>> import tensorflow as tf
    >>> x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    >>> train_output = neupy.output(x, training=True)
    >>> inference_output = neupy.output(x)

The same ``train_output`` value can be obtained with ``training_outputs`` attribute.

.. code-block:: python

    >>> train_output = network.training_outputs
    >>> inference_output = network.outputs

It's important to note, that any argument can be propagate though the network and custom layers can be designed in the way that allows to change behavior of the layer.

Access variables
----------------

Variables can be accessed with the help of the ``variables`` attribute.

.. code-block:: python

    >>> variables = network.variables
    >>> len(variables)  # number of variables

The ``variables`` attribute returns dictionary. In the dictionary, each key will be a tuple ``(layer, variable_name)`` and value will be Tensorflow's variable, associated with specified layer layer.

.. code-block:: python

    >>> for (layer, varname), variable in network.variables.items():
    ...     print(layer.name, varname, variable.shape)
    ...
    convolution-1 weight (3, 3, 1, 18)
    convolution-1 bias (18,)
    convolution-2 weight (3, 3, 18, 36)
    convolution-2 bias (36,)
    convolution-3 weight (3, 3, 36, 36)
    convolution-3 bias (36,)
    relu-4 weight (1764, 128)
    relu-4 bias (128,)
    sigmoid-1 weight (128, 10)
    sigmoid-1 bias (10,)

For some cases, it doesn't matter from which exact layer each specific variable came from. We can easily obtain list of Tensorflow variables in the following way.

.. code-block:: python

    >>> variables_only = list(network.variables.values())


Putting everything together
---------------------------

.. code-block:: python

    import tensorflow as tf
    from neupy.layers import *

    Conv = Convolution.define(padding='same')
    network = join(
        Input((28, 28, 1)),

        Conv((3, 3, 18)) >> Relu(),
        MaxPooling((2, 2)),

        Conv((3, 3, 36)) >> Relu(),
        Conv((3, 3, 36)) >> Relu(),
        MaxPooling((2, 2)),

        Reshape(),

        Relu(256) >> Dropout(0.5),
        Sigmoid(10),
    )

    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    training_output = network.output(x, training=True)
    loss = tf.reduce_mean((training_output - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(500):
            # The `iter_batches` function has to be defined by the user
            for x_batch, y_batch in iter_batches():
                training_loss = sess.run(loss, feed_dict={x: x_batch, y: y_batch})
                print('Training loss (epoch #{}): {:.6f}'.format(epoch + 1, training_loss))
