Create custom layers
====================

.. contents::

Element-wise transformation
---------------------------

The simplest type of layers is the one that doesn't modify a shape of the input value. To construct this layer we need to inherit from the ``BaseLayer`` class and define ``output`` method.

.. code-block:: python

    from neupy import layers
    from neupy.utils import asfloat

    class DoubleInput(layers.BaseLayer):
        def output(self, input_value):
            return asfloat(2) * input_value

From the code, you can see that I've used ``asfloat`` function. This function converts any number to a float. Type of the float depends on the ``theano.config.floatX`` variable. This function gives flexibility to automatically convert input variable to the float type that Tensorflow uses on the backend.

Layers with activation function
-------------------------------

Layers with activation function is a special type of layers. It has a different behavior depending on the input size. If input size is specified it applies linear transformation to the input after that pass it through the activation function. If input size wasn't presented than layer passes input directly through the activation function

.. code-block:: python

    layers.Relu()  # relu(x)
    layers.Relu(10)  # relu(W * x + b)

To be able to construct your own layer with different activation function you need to inherit from the ``ActivationLayer`` class.

.. code-block:: python

    import theano.tensor as T
    from neupy import layers

    class Squared(layers.ActivationLayer):
        def activation_function(self, input_value):
            return T.square(input_value)

Validate input shape
--------------------

Not all relations between layers are suitable. For instance, we are not able to apply pooling to the matrix. For this cases we need to have an ability to validate input shape and trigger error that will inform us about connection issues.

.. code-block:: python

    from neupy import layers
    from neupy.exceptions import LayerConnectionError

    class Pooling(layers.BaseLayer):
        def validate(self, input_shape):
            if len(input_shape) != 3:
                raise LayerConnectionError("Invalid connection")

We can use any type of exception, not only ``LayerConnectionError``.

Layer that modify input shape
-----------------------------

.. code-block:: python

    import theano.tensor as T
    from neupy import layers

    class Mean(layers.BaseLayer):
        @property
        def output_shape(self):
            # convert: (3, 28, 28) -> (28, 28)
            # convert: (10,) -> (1,)
            return self.input_shape[1:] or (1,)

        def output(self, input_value)
            return T.mean(input_value, axis=1)

Add parameters to the layer
---------------------------

.. code-block:: python

    from neupy import layers

    class Wx(layers.BaseLayer):
        def initialize(self):
            super(Wx, self).initialize()
            self.add_parameter(name='weight', shape=(10, 10),
                               value=init.Uniform(), trainable=True)

        def output(self, input_value):
            return T.dot(self.weight, input_value)

Initialization method triggers when the layer has defined input shape.

Layer that accepts multiple inputs
----------------------------------

Layers like :layer:`Concatenate` accept multiple inputs and concatenate them in one. To be able to modify multiple inputs we need to make a small modification in the ``output`` method.

.. code-block:: python

    from neupy import layers

    class SumElementwise(layers.BaseLayer):
        def output(self, *input_values):
            return sum(input_values)
