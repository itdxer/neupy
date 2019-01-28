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

    network = Input(3) >> Sigmoid(2) >> Sigmoid(1)
    optimizer = algorithms.GradientDescent(network, verbose=True)
    optimizer.fit(x_train, y_train, epochs=100)

Transform method
----------------

You can use ``transform`` method instead of the ``predict`` method.

.. code-block:: python

    from neupy import algorithms

    # Function `load_data` is not implemented
    x_train, y_train = load_data()

    network = Input(3) >> Sigmoid(2) >> Sigmoid(1)
    optimizer = algorithms.GradientDescent(network, verbose=True)
    y_predicted = optimizer.transform(x_train)

Pipelines
---------

It's possible to use NeuPy in scikit-learn pipelines.

.. code-block:: python

    from sklearn import preprocessing, pipeline
    from neupy import algorithms

    network = Input(3) >> Sigmoid(2) >> Sigmoid(1)
    pipeline = pipeline.Pipeline([
        ('min_max_scaler', preprocessing.MinMaxScaler()),
        ('backpropagation', algorithms.GradientDescent(network)),
    ])

    # Function `load_data` is not implemented
    x_train, y_train, x_test, y_test = load_data()

    pipeline.fit(x_train, y_train, backpropagation__epochs=1000)
    y_predict = pipeline.predict(x_test)

Issues
------

Not all features from scikit-learn library can be used with NeuPy. Copying of the networks and training algorithms cannot be done in a simple way and any function or class from scikit-learn that depends on the ``clone`` function will fail. For example, function like `cross_val_score <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html>`_ will not work with NeuPy classes.

Also, copying neural network might not be enough, because weights from the network will be copied as well. And cross validation on the copied network won't show you exact performance, because network has been already pre-trained before it was copied.
