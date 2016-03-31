Storage
=======

After succesful learning you can save network and later use it for prediction tasks.
There already exists awesome module - `dill <https://github.com/uqfoundation/dill>`_.
As a ``pickle`` library ``dill`` provides similar interface to serialize and de-serialize built-in python objects.
The main advantage of this is a model that can store a network class and attributes without additional functionality.

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
    ... bpnet = algorithms.Backpropagation((2, 5, 1), step=0.1, verbose=False)
    >>> bpnet.train(data, target, epochs=10000)
    >>>
    >>> predicted = bpnet.predict(data)
    >>> bpnet.error(predicted, target.reshape(target.size, 1))
    0.000756823576315
    >>>
    >>> with open('network-storage.dill', 'wb') as f:
    ...     dill.dump(bpnet, f)
    ...

And then you can use it from other file and try to reproduce the same error rate.

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
like ``dumps`` or ``loads`` are also available.
