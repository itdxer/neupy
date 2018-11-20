Storage for Neural Networks
===========================



Save and load layer parameters
------------------------------

NeuPy allows to store network's parameters in a hdf5 file.

.. code-block:: python

    from neupy import layers, storage

    network = layers.join(
        layers.Input(10),
        layers.Relu(20),
        layers.Relu(30),
        layers.Softmax(10),
    )
    storage.save(network, filepath='/path/to/file.hdf5')

To be able to load parameters you need to have predefined network structure. Using layer names NeuPy can restore parameters from the hdf5 file.

.. code-block:: python

    storage.load(network, filepath='/path/to/file.hdf5')

NeuPy supports other storage formats

.. csv-table::
    :header: "Format", "Save function", "Load function"

    "hdf5 file", ":class:`save_hdf5 <neupy.storage.save_hdf5>` (or :class:`save <neupy.storage.save>`)", ":class:`load_hdf5 <neupy.storage.load_hdf5>` (or :class:`load <neupy.storage.load>`)"
    "pickle file", ":class:`save_pickle <neupy.storage.save_pickle>`", ":class:`load_pickle <neupy.storage.load_pickle>`"
    "json file", ":class:`save_json <neupy.storage.save_json>`", ":class:`load_json <neupy.storage.load_json>`"
    "python dict", ":class:`save_dict <neupy.storage.save_dict>`", ":class:`load_dict <neupy.storage.load_dict>`"


Save and load algorithms
------------------------

After successful learning you can save network and later re-use it. You can do it with external library - `dill <https://github.com/uqfoundation/dill>`_. As a ``pickle`` library ``dill`` provides similar interface to serialize and de-serialize python objects. The main advantage of this is a functionality that can store a network class and attributes without additional modification to the classes.

First of all you need to install ``dill`` library

.. code-block:: bash

    $ pip install dill>=0.2.3

There is a simple example for network serialisation.

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

For the neural networks with fixed architectures it's possible to save and load your algorithms using ``pickle`` library.

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

Also, you can access all the parameters using ``get_params`` method (as in the scikit-learn).

.. code-block:: python

    >>> sofm.get_params()
    {'n_inputs': 2,
     'n_outputs': 4,
     'weight': array([[0.75264576, 0.26932708, 0.72538974, 0.25271294],
                      [0.75495447, 0.26936587, 0.22114073, 0.75396885]]),
     'features_grid': (4, 1)}
