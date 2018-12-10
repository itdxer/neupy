.. _surgery:

Surgery
=======

In many applications it's important to be able to access only part of the network. NeuPy supports a few methods that allow to slice network. These methods can change structure of the network or provide access to specific layers.

Network slicing
---------------

In NeuPy, it's possible to slice neural networks in order to get part of the network with different input or output layers.

.. code-block:: python

    >>> from neupy.layers import *
    >>>
    >>> network = join(
    ...     Input(10),
    ...     Relu(20, name='relu-2'),
    ...     Relu(30, name='relu-3'),
    ...     Relu(40, name='relu-4'),
    ...     Relu(50, name='relu-5'),
    ... )
    >>> network
    Input(10) > Relu(20) > Relu(30) > Relu(40) > Relu(50)

The ``end`` method can change network's output layer. For example, we want to get output from the ``relu-4`` layer instead of the ``relu-5``.

.. code-block:: python

    >>> network.end('relu-4')
    Input(10) > Relu(20) > Relu(30) > Relu(40)

The same can be done for the input layers with help of the ``start`` method.

.. code-block:: python

    >>> network.start('relu-2')
    Relu(20) > Relu(30) > Relu(40) > Relu(50)

These methods can be combined in sequence

.. code-block:: python

    >>> network.start('relu-2').end('relu-4')
    Relu(20) > Relu(30) > Relu(40)

Also, it's possible to point into multiple input and output layers

.. code-block:: python

    >>> from neupy.layers import *
    >>>
    >>> network = Input(10) > Relu(20, name='relu-2')
    >>>
    >>> output_1 = Relu(30, name='relu-3') > Sigmoid(1)
    >>> output_2 = Relu(40, name='relu-4') > Sigmoid(2)
    >>>
    >>> network = network > [output_1, output_2]
    >>>
    >>> network
    10 -> [... 6 layers ...] -> [(1,), (2,)]
    >>>
    >>> network.end('relu-3', 'relu-4')
    10 -> [... 4 layers ...] -> [(30,), (40,)]

Layer instance can be used as identifiers for the slicing method instead of the names.

.. code-block:: python

    >>> from neupy.layers import *
    >>>
    >>> input_layer = Input(10)
    >>> relu_2 = Relu(20)
    >>> relu_3 = Relu(30)
    >>>
    >>> network = input_layer > relu_2 > relu_3
    >>> network
    Input(10) > Relu(20) > Relu(30)
    >>>
    >>> network.end(relu_2)
    Input(10) > Relu(20)

.. raw:: html

    <br>

Find layer by name in the network
---------------------------------

Each name is a unique identifier for the layer inside of the network. Any layer can be accessed using the ``layer`` method.

.. code-block:: python

    >>> from neupy.layers import *
    >>>
    >>> network = join(
    ...     Input(10, name='input-1'),
    ...     Relu(8, name='relu-0'),
    ...     Relu(5, name='relu-1'),
    ... )
    >>>
    >>> network.layer('relu-0')
    Relu(8)
    >>>
    >>> network.layer('relu-1')
    Relu(5)


Exception will be triggered in case if layer with specified name wasn't defined in the network.

.. code-block:: python
    >>> network.layer('test')
    Traceback (most recent call last):
      ...
    NameError: Cannot find layer with name 'test'

.. raw:: html

    <br>
