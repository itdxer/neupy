.. _surgery:

Surgery
=======

In many applications it's important to be able to access only part of the network.

Use different input and output layers
-------------------------------------

To be able to use different input and output layers you need to use ``start`` and ``end`` methods. Here is an example.

.. code-block:: python

    >>> from neupy.layers import *
    >>>
    >>> network = join(
    ...     Input(10),
    ...     Relu(20, name='relu-2'),
    ...     Relu(30, name='relu-3'),
    ...     Relu(40, name='relu-4'),
    ...     Relu(50),
    ... )
    >>> network
    Input(10) > Relu(20) > Relu(30) > Relu(40) > Relu(50)
    >>>
    >>> network.end('relu-4')
    Input(10) > Relu(20) > Relu(30) > Relu(40)
    >>>
    >>> network.start('relu-2')
    Relu(20) > Relu(30) > Relu(40) > Relu(50)
    >>>
    >>> network.start('relu-2').end('relu-4')
    Relu(20) > Relu(30) > Relu(40)

In addition, it's possible to point into multiple input and output layers

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

Also, instead of using names we can specify layer instance

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

Each name is a unique identifier for the layer inside of the network. Any layer can be accessed using the `layer` method.

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
    >>>
    >>> network.layer('test')
    Traceback (most recent call last):
      ...
    NameError: Cannot find layer with name 'test'

.. raw:: html

    <br>
