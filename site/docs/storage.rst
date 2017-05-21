Storage for Neural Networks
===========================

Save and load layer parameters
------------------------------

NeuPy allows storing network's parameters in a pickle file. Here is an example

.. code-block:: python

    from neupy import layers, storage

    network = layers.join(
        layers.Input(10),
        layers.Relu(20),
        layers.Relu(30),
        layers.Softmax(10),
    )
    storage.save(network, filepath='/path/to/file.pickle')

To be able to load parameters you need to have predefined network structure. Using layer names NeuPy can restore parameters from the pickle file.

.. code-block:: python

    storage.load(network, filepath='/path/to/file.pickle')

Since parameters are stored in a regular pickle files it's possible to load them without NeuPy.

.. code-block:: python

    import pickle

    with open('/path/to/file.pickle', 'rb') as f:
        parameters = pickle.load(f)

The stored object in the pickle file has the following format.

.. code-block:: python

    {
        'layer-name-1': {
            'weight': np.array([...]),
            'bias': np.array([...]),
        },
        'layer-name-2': {
            'weight': np.array([...]),
            'bias': np.array([...]),
        },
        ...
    }


Save and load algorithms
------------------------

After succesful learning you can save network and later use it for prediction tasks. There already exists awesome library - `dill <https://github.com/uqfoundation/dill>`_. As a ``pickle`` library ``dill`` provides similar interface to serialize and de-serialize built-in python objects. The main advantage of this is a functionality that can store a network class and attributes without additional functionality.

First of all you need to install ``dill`` library

.. code-block:: bash

    $ pip install dill>=0.2.3

There is a simple example for network serialization.

.. code-block:: python

    >>> import dill
    >>> import numpy as np
    >>> from sklearn import datasets, preprocessing
    >>> from neupy import algorithms
    >>>
    >>> np.random.seed(0)
    >>>
    >>> # Prepare the data
    ... data, target = datasets.make_regression(n_features=2, n_targets=1)
    >>> data = preprocessing.MinMaxScaler().fit_transform(data)
    >>> target = preprocessing.MinMaxScaler().fit_transform(target)
    >>>
    >>> # Init and train network
    ... bpnet = algorithms.GradientDescent((2, 5, 1), step=0.1, verbose=False)
    >>> bpnet.train(data, target, epochs=10000)
    >>>
    >>> predicted = bpnet.predict(data)
    >>> bpnet.error(predicted, target.reshape(target.size, 1))
    0.000756823576315
    >>>
    >>> with open('network-storage.dill', 'wb') as f:
    ...     dill.dump(bpnet, f)
    ...

And then you can load it from the ``network-storage.dill`` file and try to reproduce the same error rate.

.. code-block:: python

    >>> import dill
    >>> import numpy as np
    >>> from sklearn import datasets, preprocessing
    >>>
    >>> np.random.seed(0)
    >>>
    >>> # Get the same data set because we use the same seed number.
    ... data, target = datasets.make_regression(n_features=2, n_targets=1)
    >>> data = preprocessing.MinMaxScaler().fit_transform(data)
    >>> target = preprocessing.MinMaxScaler().fit_transform(target)
    >>>
    >>> with open('network-storage.dill', 'rb') as f:
    ...     bpnet = dill.load(f)
    ...
    >>> predicted = bpnet.predict(data)
    >>> bpnet.error(predicted, target.reshape(target.size, 1))
    0.00075682357631507964

The interface for ``dill`` library is the same as for ``pickle``, so functions
like ``dumps`` or ``loads`` are available.

Save and load networks with fixed architectures
-----------------------------------------------

For the neural networks with fixed architecures it's possible to save and load your algorithms using ``pickle`` library.

.. code-block:: python

    import pickle
    from neupy import algorithms

    # Initialize and train SOFM network
    sofm = algorithms.SOFM(n_inputs=2, n_outputs=4)
    sofm.train(data)

    # Save pre-trained SOFM network
    with open('/path/to/sofm.pickle', 'wb') as f:
        pickle.dump(sofm, f)

    # Load SOFM network from the pickled file
    with open('/path/to/sofm.pickle', 'rb') as f:
        loaded_sofm = pickle.load(f)
