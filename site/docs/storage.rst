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

Data in the `parameters` variable is easely accesible.

.. code-block:: python

    >>> parameters.keys()
    ['layers', 'graph', 'metadata']
    >>> len(parameters['layers'])
    4
    >>> for layer_data in parameters['layers']:
    ...     print("Name: {}".format(layer_data['name']))
    ...     print("Available keys: {}".format(layer_data.keys()))
    ...     if layer_data['parameters']:
    ...         print("Parameters: {}".format(layer_data['parameters'].keys()))
    ...         print("Weight shape: {}".format(layer_data['parameters']['weight']['value'].shape))
    ...     print('-' * 20)
    ...
    Name: input-1
    Available keys: ['name', 'parameters', 'class_name', 'input_shape', 'configs', 'output_shape']
    --------------------
    Name: relu-1
    Available keys: ['name', 'parameters', 'class_name', 'input_shape', 'configs', 'output_shape']
    Parameters: ['bias', 'weight']
    Weight shape: (10, 20)
    --------------------
    Name: relu-2
    Available keys: ['name', 'parameters', 'class_name', 'input_shape', 'configs', 'output_shape']
    Parameters: ['bias', 'weight']
    Weight shape: (20, 30)
    --------------------
    Name: softmax-1
    Available keys: ['name', 'parameters', 'class_name', 'input_shape', 'configs', 'output_shape']
    Parameters: ['bias', 'weight']
    Weight shape: (30, 10)
    --------------------

NeuPy supports other storage formats

.. csv-table::
    :header: "Format", "Save function", "Load function"

    "pickle file", ":class:`save_pickle <neupy.storage.save_pickle>` (or :class:`save <neupy.storage.save>`)", ":class:`load_pickle <neupy.storage.load_pickle>` (or :class:`load <neupy.storage.load>`)"
    "hdf5 file", ":class:`save_hdf5 <neupy.storage.save_hdf5>`", ":class:`load_hdf5 <neupy.storage.load_hdf5>`"
    "json file", ":class:`save_json <neupy.storage.save_json>`", ":class:`load_json <neupy.storage.load_json>`"
    "python dict", ":class:`save_dict <neupy.storage.save_dict>`", ":class:`load_dict <neupy.storage.load_dict>`"


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
