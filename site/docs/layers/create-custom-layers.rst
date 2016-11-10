Create custom layers
====================

There are a few main types of layers. We are going to check them one by one in order of complexity.

Element-wise transformation
***************************

The simplest one layer that makes only element-wise transformation. This layer doesn't include any trainable parameters and it doesn't change input's shape. For simplisity we can build a layer that adds one to each elements in the input tensor.

.. code-block:: python

    from neupy import layers

    class ShiftByOneLayer(layers.BaseLayer):
        def output(self, value):
            return value + 1

``BaseLayer`` is a main class that contains all important methods and attributes that any layer requires.

Layers with activation function
*******************************

If you want to have a layer that behaves like any other built-in layer with activation function in the NeuPy, you should inherit your class from the ``ActivationLayer`` class. Let's consider and example of layer with a square activation function.

.. code-block:: python

    import theano.tensor as T
    from neupy import layers

    class SquareLayer(layers.ActivationLayer):
        def activation_function(self, input_value):
            return T.square(input_value)

Advanced methods
****************

There are a few other methods that can be useful in case of more complecated layers.

Some layers makes an output shape modification. If you don't define the new output shape in the layer class you will not be able to attach layer to the other in the network. Here is an example.

.. code-block:: python

    from neupy import layers

    class FeatureMeanLayer(layers.BaseLayer):
        @property
        def output_shape(self):
            return (1,)

        def output(self, value):
            return value.mean(axis=1)

The ``FeatureMeanLayer`` layer makes a feature transformation and modifies input matrix shape. We define ``output_shape`` property as an additional layer's parameter. Now other layers in the sequence know the expected layers output shape and adjust their parameters to the correct shape.

The other important method is an ``initialize``. The main purpose of this method is to apply all necesary initializations related to the network. For instance, random weight or bias parameters. You can't define them in the ``__init__`` method, because when you create layer instance full network structure is unknown and you don't have all the necessary information. Let's extend the previous layer with the ``initialize`` method.

.. code-block:: python

    from neupy import layers
    from neupy.utils import asfloat

    class FeatureMeanLayer(layers.BaseLayer):
        def initialize(self):
            super(FeatureMeanLayer, self).initialize()

            self.scaler = theano.shared(
                name="{}/scaler".format(self.name),
                value=asfloat(1)
            )
            self.parameters.append(self.scaler)

        @property
        def output_shape(self):
            return (1,)

        def output(self, value):
            return self.scaler * value.mean(axis=1)

In this example I've added a few other feature to consider a couple of useful attributes and functions. Let's check them one by one. ``self.name`` defines layer's identifier as an integer number. This number is basically a layer's index number in the sequence. The ``self.parameters`` attribute is a list that contains all trainable parameters. Usually it is a weight or bias, but you can define any parameter you want. The ``asfloat`` function just converts any number to the float number. The type of the float number depends on the Theano's ``theano.config.floatX`` variable.
