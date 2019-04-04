Replicate layers and networks
=============================

Copy layers/networks
--------------------

Any layer or network could be copied. For example, we can create layer and copy it as many times as we want.

.. code-block:: python

    >>> from neupy.layers import *
    >>> layer = layers.Relu(10)
    >>> layer
    Relu(10, alpha=0, weight=HeNormal(gain=2), bias=Constant(0), name='relu-1')
    >>>
    >>> import copy
    >>> copy.copy(layer)
    Relu(10, alpha=0, weight=HeNormal(gain=2), bias=Constant(0), name='relu-2')
    >>>
    >>> copy.copy(layer)
    Relu(10, alpha=0, weight=HeNormal(gain=2), bias=Constant(0), name='relu-3')

Each new copy gets the same set of parameters. Also, notice that each new copy get new name. It works because we
didn't specify exact name of the layer.

.. code-block:: python

    >>> layer = Relu(10, name='relu-layer')
    >>> layer
    Relu(10, alpha=0, weight=HeNormal(gain=2), bias=Constant(0), name='relu-layer')
    >>>
    >>> copy.copy(layer)
    Relu(10, alpha=0, weight=HeNormal(gain=2), bias=Constant(0), name='relu-layer')

In order to create layers with custom name, we can specify name as a string that could be formatted with a unique
index value.

.. code-block:: python

    >>> layer = Relu(10, name='relu-layer-{}')
    >>> layer
    Relu(10, alpha=0, weight=HeNormal(gain=2), bias=Constant(0), name='relu-layer-1')
    >>>
    >>> copy.copy(layer)
    Relu(10, alpha=0, weight=HeNormal(gain=2), bias=Constant(0), name='relu-layer-2')

All the examples above also work for the networks

.. code-block:: python

    >>> block = Conv((3, 3, 32)) >> Relu() >> BN()
    >>> block
    <unknown> -> [... 3 layers ...] -> (?, ?, ?, 32)
    >>>
    >>> copy.copy(block)
    <unknown> -> [... 3 layers ...] -> (?, ?, ?, 32)

Repeat the same layer/network multiple times
--------------------------------------------

Certain neural network architectures might have single layer or group of layers repeated multiple times in a sequence. For example:

.. code-block:: python

    >>> from neupy.layers import *
    >>> network = Input(20) >> Relu(10) >> Relu(10) >> Relu(10) >> Relu(10)
    >>> network
    (?, 20) -> [... 5 layers ...] -> (?, 10)

In order to avoid redundant repetitions, NeuPy introduced function called ``repeat``. This function copies layer
multiple times and connects original and all copied layers into a sequence. We can rewrite previous example, using
``repeat`` function in order to get the exactly the same neural network architecture.

.. code-block:: python

    >>> network = Input(20) >> repeat(Relu(10), n=4)
    >>> network
    (?, 20) -> [... 5 layers ...] -> (?, 10)

And the same function will work if applied to the network.

.. code-block:: python

    >>> block = Conv((3, 3, 32)) >> Relu() >> BN()
    >>> block
    <unknown> -> [... 3 layers ...] -> (?, ?, ?, 32)
    >>>
    >>> repeat(block, n=5)
    <unknown> -> [... 15 layers ...] -> (?, ?, ?, 32)

It's important to remember that input shape of the layer/network should be compatible with it's output shape.
Otherwise exception will be triggered.

Caveats
-------

Copying and repetition make more sense when layer hasn't been initialized yet. Let's check the following example:

.. code-block:: python

    >>> from neupy.layers import *
    >>> layer = layers.Relu(10)
    >>> layer
    Relu(10, alpha=0, weight=HeNormal(gain=2), bias=Constant(0), name='relu-1')

We can see that ``weight`` and ``bias`` hasn't been generated yet. We can add layer to the network and create variables for it.

.. code-block:: python

    >>> network = Input(20) >> layer
    >>> network.create_variables()
    >>> layer
    Relu(10, alpha=0, weight=<Variable shape=(20, 10)>, bias=<Variable shape=(10,)>, name='relu-1')

We can see that now each parameter of the layer has it's own variable. If we try to copy layer with initialized
variables that's what we will get.

.. code-block:: python

    >>> copy.copy(layer)
    Relu(10, alpha=0, weight=<Array shape=(20, 10)>, bias=<Array shape=(10,)>, name='relu-2')

Now each parameter has it's value specified as an array, which is just a copy of the value stored in the original
variable. For this layer, variables hasn't been created yet, since it's not a part of any network.