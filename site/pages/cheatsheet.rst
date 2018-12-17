.. _cheat-sheet:

Cheat sheet
===========

.. contents::
    :depth: 2

Algorithms
**********

Algorithms based on backpropagation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _cheatsheet-backprop-algorithms:

Training algorithms
+++++++++++++++++++

.. autosummary::
    :toctree: ../modules/generated/
    :template: autosummary/class.rst

   neupy.algorithms.Momentum
   neupy.algorithms.GradientDescent
   neupy.algorithms.Adam
   neupy.algorithms.Adamax
   neupy.algorithms.RMSProp
   neupy.algorithms.Adadelta
   neupy.algorithms.Adagrad
   neupy.algorithms.ConjugateGradient
   neupy.algorithms.QuasiNewton
   neupy.algorithms.LevenbergMarquardt
   neupy.algorithms.Hessian
   neupy.algorithms.HessianDiagonal
   neupy.algorithms.RPROP
   neupy.algorithms.IRPROPPlus

Regularization methods
++++++++++++++++++++++

.. autosummary::
    :toctree: ../modules/generated/
    :template: autosummary/class.rst

    neupy.algorithms.WeightDecay
    neupy.algorithms.WeightElimination
    neupy.algorithms.MaxNormRegularization

Learning rate update rules
++++++++++++++++++++++++++

.. code-block:: python

    from neupy import algorithms
    from neupy.layers import *

    optimizer = algorithms.Momentum(
        Input(5) > Relu(10) > Sigmoid(1),
        step=algorithms.step_decay(
            initial_value=0.1,
            reduction_freq=100,
        )
    )

.. autosummary::
    :toctree: ../modules/generated/
    :template: autosummary/function.rst
    :nosignatures:

    neupy.algorithms.step_decay
    neupy.algorithms.exponential_decay
    neupy.algorithms.polynomial_decay

Neural Networks with Radial Basis Functions (RBFN)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../modules/generated/
    :template: autosummary/class.rst

    neupy.algorithms.GRNN
    neupy.algorithms.PNN
    neupy.algorithms.RBFKMeans

Autoasociative Memory
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../modules/generated/
    :template: autosummary/class.rst

    neupy.algorithms.DiscreteBAM
    neupy.algorithms.CMAC
    neupy.algorithms.DiscreteHopfieldNetwork

Competitive Networks
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../modules/generated/
    :template: autosummary/class.rst

    neupy.algorithms.ART1
    neupy.algorithms.GrowingNeuralGas
    neupy.algorithms.SOFM
    neupy.algorithms.LVQ
    neupy.algorithms.LVQ2
    neupy.algorithms.LVQ21
    neupy.algorithms.LVQ3

Associative
~~~~~~~~~~~

.. autosummary::
    :toctree: ../modules/generated/
    :template: autosummary/class.rst

    neupy.algorithms.Oja
    neupy.algorithms.Kohonen
    neupy.algorithms.Instar
    neupy.algorithms.HebbRule

Boltzmann Machine
~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../modules/generated/
    :template: autosummary/class.rst

    neupy.algorithms.RBM

Layers
******

.. code-block:: python

    from neupy.layers import *
    network = Input(32) > Relu(16) > Softmax(10)

Layers with activation function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../modules/generated/
    :template: autosummary/class.rst

    neupy.layers.Linear
    neupy.layers.Sigmoid
    neupy.layers.HardSigmoid
    neupy.layers.Tanh
    neupy.layers.Relu
    neupy.layers.LeakyRelu
    neupy.layers.Elu
    neupy.layers.PRelu
    neupy.layers.Softplus
    neupy.layers.Softmax

Convolutional layers
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../modules/generated/
    :template: autosummary/class.rst

    neupy.layers.Convolution
    neupy.layers.Deconvolution


Recurrent layers
~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../modules/generated/
    :template: autosummary/class.rst

    neupy.layers.LSTM
    neupy.layers.GRU

Pooling layers
~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../modules/generated/
    :template: autosummary/class.rst

    neupy.layers.MaxPooling
    neupy.layers.AveragePooling
    neupy.layers.Upscale
    neupy.layers.GlobalPooling

Normalization layers
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../modules/generated/
    :template: autosummary/class.rst

    neupy.layers.BatchNorm
    neupy.layers.LocalResponseNorm

Stochastic layers
~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../modules/generated/
    :template: autosummary/class.rst

    neupy.layers.Dropout
    neupy.layers.GaussianNoise

Merge layers
~~~~~~~~~~~~

.. autosummary::
    :toctree: ../modules/generated/
    :template: autosummary/class.rst

    neupy.layers.Elementwise
    neupy.layers.Concatenate
    neupy.layers.GatedAverage

Other layers
~~~~~~~~~~~~

.. autosummary::
    :toctree: ../modules/generated/
    :template: autosummary/class.rst

    neupy.layers.Input
    neupy.layers.Reshape
    neupy.layers.Transpose
    neupy.layers.Embedding

Architectures
*************

.. code-block:: python

    >>> from neupy import architectures
    >>> resnet = architectures.resnet50()
    >>> resnet
    (224, 224, 3) -> [... 187 layers ...] -> 1000

.. autosummary::
    :toctree: ../modules/generated/
    :template: autosummary/function.rst
    :nosignatures:

    neupy.architectures.vgg16
    neupy.architectures.vgg19
    neupy.architectures.squeezenet
    neupy.architectures.resnet50
    neupy.architectures.mixture_of_experts

.. _init-methods:

Parameter initialization
************************

.. code-block:: python

    from neupy.init import *
    from neupy.layers import *
    from neupy import algorithms

    gdnet = algorithms.GradientDescent([
          Input(784),
          Relu(100, weight=HeNormal(), bias=Constant(0)),
          Softmax(10, weight=Uniform(-0.01, 0.01)),
    ])

.. raw:: html

    <br>

.. autosummary::
    :toctree: ../modules/generated/
    :template: autosummary/class.rst

    neupy.init.Constant
    neupy.init.Normal
    neupy.init.Uniform
    neupy.init.Orthogonal
    neupy.init.HeNormal
    neupy.init.HeUniform
    neupy.init.XavierNormal
    neupy.init.XavierUniform

Datasets
********

.. autosummary::
    :toctree: ../modules/generated/
    :template: autosummary/function.rst
    :nosignatures:

    neupy.datasets.load_digits
    neupy.datasets.make_digits
    neupy.datasets.make_reber
    neupy.datasets.make_reber_classification
