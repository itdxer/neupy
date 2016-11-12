Connections between layers
--------------------------

Any connection between layers in NeuPy is a `Directional Acyclic Graph (DAG) <https://en.wikipedia.org/wiki/Directed_acyclic_graph>`_.

Feedforward connections
=======================

Feedforward connection is a very simple graph where all layers connected in sequential order.

Multilayer Perceptron
*********************

.. code-block:: python

    from neupy import layers

    feedforward = layers.join(
        layers.Input(784),
        layers.Relu(500),
        layers.Relu(300),
        layers.Softmax(10),
    )

.. figure:: images/feedforward-graph-connection.png
    :align: center
    :alt: Feedforward connections in NeuPy

Convolutional Neural Networks (CNN)
***********************************

.. code-block:: python

    from neupy import layers

    convnet = layers.join(
        layers.Input((3, 28, 28)),

        layers.Convolution((32, 3, 3)) > layers.Relu(),
        layers.Convolution((48, 3, 3)) > layers.Relu(),
        layers.MaxPooling((2, 2)),

        layers.Reshape(),
        layers.Softmax(10),
    )

.. figure:: images/conv-graph-connection.png
    :align: center
    :alt: Feedforward convolutional connections in NeuPy

Graph connections
=================

.. code-block:: python

    from neupy import layers

    connection = layers.join(
        layers.Input((3, 10, 10)),
        layers.Parallel(
            [[
                layers.Convolution((32, 3, 3)),
                layers.Relu(),
                layers.MaxPooling((2, 2)),
            ], [
                layers.Convolution((16, 7, 7)),
                layers.Relu(),
            ]],
            layers.Concatenate()
        ),
        layers.Reshape(),
        layers.Softmax(10),
    )

.. figure:: images/conv-parallel-connection.png
    :align: center
    :alt: Graph connections in NeuPy

Prepared input from hidden layers
=================================

test_dict_based_inputs_into_connection

Visualize connections
=====================

For the debugging it's useful to explore connection's structure. It's possible to create graph visualization in NeuPy.

.. code-block:: python

    from neupy import layers, plots

    connection = layers.join(
        layers.Input((3, 10, 10)),
        layers.Parallel(
            [[
                layers.Convolution((32, 3, 3)),
                layers.Relu(),
                layers.MaxPooling((2, 2)),
            ], [
                layers.Convolution((16, 7, 7)),
                layers.Relu(),
            ]],
            layers.Concatenate()
        ),
        layers.Reshape(),
        layers.Softmax(10),
    )
    plots.layer_structure(connection)
