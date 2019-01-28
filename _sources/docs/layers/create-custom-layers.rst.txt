Create custom layers
====================

.. contents::

Element-wise transformation
---------------------------

The simplest type of layers is the one that doesn't modify a shape of the input value. In order to construct this layer, we need to inherit from the ``Identity`` class and define the ``output`` method.

.. code-block:: python

    from neupy.layers import *

    class Double(Identity):
        def output(self, input, **kwargs):
            return 2 * input

Now this layer can be used in the network.

.. code-block:: python

    >>> Input(10) >> Double()
    (?, 10) -> [... 2 layers ...] -> (?, 10)

Notice that output expects ``**kwargs`` argument as well. NeuPy allows to propagate extra information through the network in the form of an argument. For example, variable ``training`` might be passed through every ``output`` method in order to control behavior of the network.

Layers with activation function
-------------------------------

Layers with activation function is a special type of layers. It can behave in two different ways depending on the initialization. If output size is specified it applies linear transformation and then applies activation function. If output size wasn't specified than layer passes input directly through the activation function.

.. code-block:: python

    layers.Relu()  # relu(x)
    layers.Relu(10)  # relu(W * x + b)

To be able to construct your own layer with custom activation function you need to inherit from the ``Linear`` layer class and specify the ``activation_function`` method.

.. code-block:: python

    import tensorflow as tf
    from neupy.layers import *

    class Square(Linear):
        def activation_function(self, input):
            return tf.square(input)

Also, notice that in this example we use **Tensorflow**. NeuPy uses Tensorflow as a computational backend for the constructible neural networks and we need to specify all operations using functions from the Tensorflow.

Layer that modifies input shape
-------------------------------

Layers that apply transformations that can modify shape of the input should be build on top of the base class for all layers, called ``BaseLayer``.

.. code-block:: python

    import tensorflow as tf
    from neupy.layers import *

    class Mean(BaseLayer):
        def output(self, input, **kwargs):
            return tf.reduce_mean(input, axis=-1, keepdims=True)

The problem with this approach is that we don't know in advance what transformation to the input's shape this layer has applied.

.. code-block:: python

    >>> Input((10, 10, 2)) >> Mean()
    (?, 10, 10, 2) -> [... 2 layers ...] -> <unknown>

The only case when it's a problem is when one of the subsequent layer might depend on the expected input shapes values. For example, when we want to initialize weights for one of the layers, expected input shape will be important information. In order to add this information to the layer we can add extra method, called ``get_output_shape``.

.. code-block:: python

    class Mean(BaseLayer):
        def get_output_shape(self, input_shape):
            # Input and output shapes from the layer has to be an instance
            # of the TensorShape class provided by tensorflow library.
            input_shape = tf.TensorShape(input_shape)
            output_shape = input_shape[:-1].concatenate(1)
            return output_shape

        def output(self, input, **kwargs):
            return tf.reduce_mean(input, axis=-1, keepdims=True)

.. code-block:: python

    >>> Input((10, 10, 2)) >> Mean()
    (?, 10, 10, 2) -> [... 2 layers ...] -> (?, 10, 10, 1)
    >>> Input((10, 10, 18)) >> Mean()
    (?, 10, 10, 18) -> [... 2 layers ...] -> (?, 10, 10, 1)

Layer that accepts multiple inputs
----------------------------------

Layers like :layer:`Concatenate` accept multiple inputs and it combines them into single tensor. To be able to modify multiple inputs we need specify fixed set of expected input variables or as undefined.

.. code-block:: python

    import tensorflow as tf
    from neupy.layers import *
    from neupy.exceptions import LayerConnectionError

    class Multiply(BaseLayer):
        def get_output_shape(self, *input_shapes):
            first_shape = input_shapes[0]

            for shape in input_shapes:
                if not shape.is_compatible_with(first_shape):
                    raise LayerConnectionError("Invalid inputs")

            return first_shape

        def output(self, *inputs, **kwargs):
            return reduce(tf.multiply, inputs)

Notice that we also added exception in case if there is something wrong with input connections. The ``get_output_shape`` method triggers each time layer added to the network, so it's possible that one of the inputs hasn't been defined yet.

.. code-block:: python

    >>> (Input((10, 10, 2)) | Input((10, 10, 2))) >> Multiply()
    [(?, 10, 10, 2), (?, 10, 10, 2)] -> [... 3 layers ...] -> (?, 10, 10, 2)
    >>>
    >>> (Input((10, 10, 2)) | Relu()) >> Multiply()
    [(?, 10, 10, 2), <unknown>] -> [... 3 layers ...] -> (?, 10, 10, 2)
    >>>
    >>> (Input((10, 10, 2)) | Input((10, 10, 4))) >> Multiply()
    ...
    LayerConnectionError: Invalid inputs...

Add parameters to the layer
---------------------------

In case if layer requires to have parameters the ``create_variables`` method has to be specified.

Some layers might have parameters that has to be trained. For example, we can create layer that multiples input be some matrix ``W``.

.. code-block:: python

    import tensorflow as tf
    from neupy import init
    from neupy.layers import *

    class Wx(BaseLayer):
        def __init__(self, outsize, name=None):
            self.outsize = outsize
            super(Wx, self).__init__(name=name)

        def get_output_shape(self, input_shape):
            n_samples, n_input_features = input_shape
            return tf.TensorShape([n_samples, self.outsize])

        def create_variables(self, input_shape):
            _, n_input_features = input_shape
            self.weight = self.variable(
                name='weight',

                # By default, we assume that every input will have 10
                # features, but in perfect case input and output shapes
                # might be parameterized by the user.
                shape=(n_input_features.value, self.outsize),

                # Default initialization method for parameters. It can
                # be pre-generated matrix or tensorflow's variables instead
                # of initializer.
                value=init.Uniform(),

                # Make sure that parameter will be learned during the
                # training. The ``False`` value means that we won't tune
                # it during backpropagation.
                trainable=True,
            )

        def output(self, input, **kwargs):
            return tf.matmul(self.weight, input)

.. code-block:: python

    >>> network = Input(5) >> Wx(10)
    >>> network
    (?, 5) -> [... 2 layers ...] -> (?, 10)

The ``self.variable`` method not only creates variable, but it also registers variable as network's parameter.

    >>> list(network.variables.values())
    [<tf.Variable 'layer/wx-5/weight:0' shape=(5, 10) dtype=float32_ref>]
