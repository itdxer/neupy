Create custom layers
====================

.. contents::

Element-wise transformation
---------------------------

The simplest type of layers is the one that doesn't modify a shape of the input value. In order to construct this layer we need to inherit from the ``BaseLayer`` class and define the ``output`` method.

.. code-block:: python

    from neupy import layers

    class Double(layers.BaseLayer):
        def output(self, input_value):
            return 2 * input_value

Now this layer can be used in the network.

.. code-block:: python

    layers.Input(10) > Double()

Layers with activation function
-------------------------------

Layers with activation function is a special type of layers. It can behave in two different ways depending on the initialization. If output size is specified it applies linear transformation and then applies activation function. If output size wasn't specified than layer passes input directly through the activation function.

.. code-block:: python

    layers.Relu()  # relu(x)
    layers.Relu(10)  # relu(W * x + b)

To be able to construct your own layer with custom activation function you need to inherit from the ``ActivationLayer`` class and specify the ``activation_function`` method.

.. code-block:: python

    import tensorflow as tf
    from neupy import layers

    class Square(layers.ActivationLayer):
        def activation_function(self, input_value):
            return tf.square(input_value)

Also, notice that in this example we use **Tensorflow**. NeuPy uses Tensorflow as a computational backend for the constructible neural networks and we need to specify all operations using functions from the Tensorflow.

Layer that modify input shape
-----------------------------

Layers with activation function can apply linear transformation and change output shape of the matrix. All the information can be derived from the input layer and specified output size. In other cases in order to work with layers that modify shape of the input tensor we need to define one extra property called ``output_shape``.

In the example below, we define layer that calculate mean of the input over last dimension.

.. code-block:: python

    import tensorflow as tf
    from neupy import layers

    class Mean(layers.BaseLayer):
        @property
        def output_shape(self):
            # converts: (28, 28, 1) -> (28, 28, 1)
            # converts: (10,) -> (1,)
            return self.input_shape[:-1] or (1,)

        def output(self, input_value):
            return tf.reduce_mean(input_value, axis=-1)

Notice from the example that we can access the ``input_shape`` property. This property derived from the layer that has been attached to this layer. It's not always available since NeuPy allows arbitrary order of the layers during the definition this information might not be available and it might be useful to add extra check that ensures that information about input shape is available and return ``None`` otherwise.

.. code-block:: python

    class Mean(layers.BaseLayer):
        @property
        def output_shape(self):
            if not self.input_shape:
                return None

            # converts: (28, 28, 1) -> (28, 28, 1)
            # converts: (10,) -> (1,)
            return self.input_shape[:-1] or (1,)

Layer that accepts multiple inputs
----------------------------------

Layers like :layer:`Concatenate` accept multiple inputs and combine them into single tensor. To be able to modify multiple inputs we need to make a small modification in the ``output`` method. We can create layer that concatenate it's inputs over last dimension.

.. code-block:: python

    import copy
    import tensorflow as tf
    from neupy import layers

    class Concatenate(layers.BaseLayer):
        axis = -1

        @property
        def output_shape(self):
            if self.input_shape:
                # With copy function we make sure the any modification to
                # the list won't effect original list of shapes.
                input_shapes = copy.copy(self.input_shape)
                output_shape = list(input_shapes.pop(0))

                for input_shape in input_shapes:
                    output_shape[self.axis] += input_shape[self.axis]

                return tuple(output_shape)

        def output(self, *input_values):
            return tf.concat(input_values, axis=self.axis)

Notice from the example that we use the ``input_shape`` property as a list. This property stores shapes from each input layer.

Validate input shape
--------------------

Not all relations between layers are suitable. For instance, we are not able to apply max pooling to the matrix. For this cases we need to have an ability to validate input shape and trigger error that will inform us about the problem.

.. code-block:: python

    from neupy import layers
    from neupy.exceptions import LayerConnectionError

    class Pooling(layers.BaseLayer):
        def validate(self, input_shape):
            if len(input_shape) != 3:
                raise LayerConnectionError("Invalid connection")

We can use any type of exception, not only ``LayerConnectionError``.

Add parameters to the layer
---------------------------

Some layers might have parameters that has to be trained. For example, we can create layer that multiples input be some matrix ``W``.

.. code-block:: python

    import tensorflow as tf
    from neupy import layers

    class Wx(layers.BaseLayer):
        def initialize(self):
            super(Wx, self).initialize()
            n_input_features = self.input_shape[0]

            self.add_parameter(
                name='weight',

                # By default, we assume that every input will have 10
                # features, but in perfect case input and output shapes
                # might be parameterized by the user.
                shape=(n_input_features, 10),

                # Default initialization method for parameters. It can
                # be pre-generated matrix instead of initializer.
                value=init.Uniform(),

                # Make sure that parameter will be learned during the
                # training. The ``False`` value means that we won't tune
                # it during backpropagation.
                trainable=True,
            )

        def output(self, input_value):
            return tf.matmul(self.weight, input_value)

Initialization method triggers when the layer receives input shape from the layer that has been attached to it.
