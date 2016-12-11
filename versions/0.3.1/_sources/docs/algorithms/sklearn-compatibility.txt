Scikit-learn compatibility
==========================

NeuPy contains a few compatibilities that make it possible use NeuPy with the scikit-learn library.

Fit method
----------

You can use ``fit`` method instead of the ``train`` method.

.. code-block:: python

    from neupy import algorithms

    # Function `load_data` is not implemented
    x_train, y_train = load_data()

    bpnet = algorithms.GradientDescent((2, 3 1))
    bpnet.fit(x_train, y_train, epochs=100)

Transform method
----------------

You can use ``transform`` method instead of the ``predict`` method.

.. code-block:: python

    from neupy import algorithms

    # Function `load_data` is not implemented
    x_train, y_train = load_data()

    bpnet = algorithms.GradientDescent((2, 3 1))
    y_predicted = bpnet.transform(x_train)

Pipelines
---------

It's possible to use NeuPy in scikit-learn pipelines.

.. code-block:: python

    from sklearn import preprocessing, pipeline
    from neupy import algorithms

    pipeline = pipeline.Pipeline([
        ('min_max_scaler', preprocessing.MinMaxScaler()),
        ('backpropagation', algorithms.GradientDescent((2, 3, 1))),
    ])

    # Function `load_data` is not implemented
    x_train, y_train, x_test, y_test = load_data()

    pipeline.fit(x_train, y_train, backpropagation__epochs=1000)
    y_predict = pipeline.predict(x_test)
