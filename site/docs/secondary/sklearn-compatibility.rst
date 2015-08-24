Scikit-learn compatibility
--------------------------

There are also a lot of good stuff in scikit-learn that are also can be useful for neural network algorithms.
NeuralPy contains a few compatibilities for making possible interaction with some scikit-learn features.

First of all instead of ``train`` method you can use ``fit``.

.. code-block:: python

    from neuralpy import algorithms

    bpnet = algorithms.Backpropagation((2, 3 1))
    bpnet.fit(x_train, y_train, epochs=100)

Also you can use scikit-learn pipelines with NeuralPy.

.. code-block:: python

    from sklearn import preprocessing, pipeline
    from neuralpy import algorithms

    pipeline = pipeline.Pipeline([
        ('min_max_scaler', preprocessing.MinMaxScaler()),
        ('backpropagation', algorithms.Backpropagation((2, 3, 1))),
    ])

    pipeline.fit(x_train, y_train, backpropagation__epochs=1000)
    y_predict = pipeline.predict(x_test)
