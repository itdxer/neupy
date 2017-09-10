.. _cheat-sheet:

Cheat sheet
===========

.. contents::

Algorithms
**********

Algorithms that use Backpropagation training approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _cheatsheet-backprop-algorithms:

Trainig algorithms
++++++++++++++++++

.. csv-table::
    :header: "Class name", "Name"

    :network:`GradientDescent`, Classic Gradient Descent
    :network:`MinibatchGradientDescent`, Mini-batch Gradient Descent
    :network:`ConjugateGradient`, Conjugate Gradient
    :network:`QuasiNewton`, quasi-Newton
    :network:`LevenbergMarquardt`, Levenberg-Marquardt
    :network:`Hessian`, Hessian
    :network:`HessianDiagonal`, Hessian diagonal
    :network:`Momentum`, Momentum
    :network:`RPROP`, RPROP
    :network:`IRPROPPlus`, iRPROP+
    :network:`Quickprop`, Quickprop
    :network:`Adadelta`, Adadelta
    :network:`Adagrad`, Adagrad
    :network:`RMSProp`, RMSProp
    :network:`Adam`, Adam
    :network:`Adamax`, AdaMax

Regularization methods
++++++++++++++++++++++

.. csv-table::
    :header: "Class name", "Name"

    :network:`WeightDecay`, Weight decay
    :network:`WeightElimination`, Weight elimination
    :network:`MaxNormRegularization`, Max-norm regularization

Learning rate update rules
++++++++++++++++++++++++++

.. csv-table::
    :header: "Class name", "Name"

    :network:`LeakStepAdaptation`, Leak Step Adaptation
    :network:`ErrDiffStepUpdate`, Error difference Update
    :network:`LinearSearch`, Linear search by Golden Search or Brent
    :network:`SearchThenConverge`, Search than converge
    :network:`StepDecay`, Minimize step monotonically after each epoch

Ensembles
~~~~~~~~~

.. csv-table::
    :header: "Class name", "Name"

    :architecture:`mixture_of_experts`, Mixture of Experts
    :network:`DynamicallyAveragedNetwork`, Dynamically Averaged Network (DAN)

Neural Networks with Radial Basis Functions (RBFN)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
    :header: "Class name", "Name"

    :network:`GRNN`, Generalized Regression Neural Network (GRNN)
    :network:`PNN`, Probabilistic Neural Network (PNN)
    :network:`RBFKMeans`, Radial basis function K-means

Autoasociative Memory
~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
    :header: "Class name", "Name"

    :network:`DiscreteBAM`, Discrete BAM Network
    :network:`CMAC`, CMAC Network
    :network:`DiscreteHopfieldNetwork`, Discrete Hopfield Network

Competitive Networks
~~~~~~~~~~~~~~~~~~~~

.. csv-table::
    :header: "Class name", "Name"

    :network:`ART1`, Adaptive Resonance Theory (ART1) Network
    :network:`SOFM`, Self-Organizing Feature Map (SOFM or SOM)
    :network:`LVQ`, Learning Vector Quantization (LVQ)
    :network:`LVQ2`, Learning Vector Quantization 2 (LVQ2)
    :network:`LVQ21`, Learning Vector Quantization 2.1 (LVQ2.1)
    :network:`LVQ3`, Learning Vector Quantization 3 (LVQ3)

Linear networks
~~~~~~~~~~~~~~~

.. csv-table::
    :header: "Class name", "Name"

    :network:`Perceptron`, Perceptron
    :network:`LMS`, LMS Network
    :network:`ModifiedRelaxation`, Modified Relaxation Network

Associative
~~~~~~~~~~~

.. csv-table::
    :header: "Class name", "Name"

    :network:`Oja`, OJA
    :network:`Kohonen`, Kohonen
    :network:`Instar`, Instar
    :network:`HebbRule`, Hebbian Neural Network

Boltzmann Machine
~~~~~~~~~~~~~~~~~

.. csv-table::
    :header: "Class name", "Name"

    :network:`RBM`, Boolean/Bernoulli Restricted Boltzmann Machine

Layers
******

Layers with activation function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
    :header: "Class name", "Description"

    ":layer:`Linear`", "Layer with linear activation function."
    ":layer:`Sigmoid`", "Layer with sigmoid activation function."
    ":layer:`HardSigmoid`", "Layer with hard sigmoid activation function."
    ":layer:`Step`", "Layer with step activation function."
    ":layer:`Tanh`", "Layer with tanh activation function."
    ":layer:`Relu`", "Layer with ReLu activation function."
    ":layer:`LeakyRelu`", "Layer with Leaky ReLu activation function."
    ":layer:`Elu`", "Layer with ELU activation function."
    ":layer:`PRelu`", "Layer with Parametric ReLu activation function."
    ":layer:`Softplus`", "Layer with softplus activation function."
    ":layer:`Softmax`", "Layer with softmax activation function."

Convolutional layers
~~~~~~~~~~~~~~~~~~~~

.. csv-table::
    :header: "Class name", "Description"

    ":layer:`Convolution`", "Convolutional layer"


Recurrent layers
~~~~~~~~~~~~~~~~

.. csv-table::
    :header: "Class name", "Description"

    ":layer:`LSTM`", "Long-Short Term Memory (LSTM) layer"
    ":layer:`GRU`", "Gated Recurrent Unit (GRU) layer"

Pooling layers
~~~~~~~~~~~~~~

.. csv-table::
    :header: "Class name", "Description"

    ":layer:`MaxPooling`", "Maximum pooling layer"
    ":layer:`AveragePooling`", "Average pooling layer"
    ":layer:`Upscale`", "Upscale layer"
    ":layer:`GlobalPooling`", "Global pooling layer"

Normalization layers
~~~~~~~~~~~~~~~~~~~~

.. csv-table::
    :header: "Class name", "Description"

    ":layer:`BatchNorm`", "Batch normalization layer"
    ":layer:`LocalResponseNorm`", "Local Response Normalization layer"

Stochastic layers
~~~~~~~~~~~~~~~~~

.. csv-table::
    :header: "Class name", "Description"

    ":layer:`Dropout`", "Dropout layer"
    ":layer:`GaussianNoise`", "Add gaussian noise to the input"

Merge layers
~~~~~~~~~~~~

.. csv-table::
    :header: "Class name", "Description"

    ":layer:`Elementwise`", "Merge multiple input layers in one with elementwise function"
    ":layer:`Concatenate`", "Concatenate multiple input layers in one based on the specified axes."
    ":layer:`GatedAverage`", "Average multiple layers based on the output from the gate layer."

Other layers
~~~~~~~~~~~~

.. csv-table::
    :header: "Class name", "Description"

    ":layer:`Input`", "Layer defines input value's feature shape"
    ":layer:`Reshape`", "Reshape tensor input"
    ":layer:`Embedding`", "Embedding layer accepts indeces as an input and returns rows from the weight matrix associated with these indeces."

Architectures
*************

.. csv-table::
    :header: "Class name", "Description"

    ":architecture:`vgg16`", "VGG16 network"
    ":architecture:`vgg19`", "VGG19 network"
    ":architecture:`squeezenet`", "SqueezeNet network"
    ":architecture:`alexnet`", "AlexNet network"
    ":architecture:`mixture_of_experts`", Mixture of Experts


.. _init-methods:

Parameter Initialization Methods
********************************

.. code-block:: python

    from neupy import algorithms, layers, init

    gdnet = algorithms.GradientDescent(
        [
            layers.Input(784),
            layers.Relu(100, weight=init.HeNormal(), bias=init.HeNormal()),
            layers.Softmax(10, weight=init.Uniform(-0.01, 0.01)),
        ]
    )

.. raw:: html

    <br>

.. csv-table::
    :header: "Class name", "Description"

    ":class:`Constant <neupy.init.Constant>`", "Initialize weights with constant values"
    ":class:`Normal <neupy.init.Normal>`", "Sample weights from the Normal distribution"
    ":class:`Uniform <neupy.init.Uniform>`", "Sample weights from the Uniformal distribution"
    ":class:`Orthogonal <neupy.init.Orthogonal>`", "Initialize matrix with orthogonal basis"
    ":class:`HeNormal <neupy.init.HeNormal>`", "Kaiming He parameter initialization method based on the Normal distribution."
    ":class:`HeUniform <neupy.init.HeUniform>`", "Kaiming He parameter initialization method based on the Uniformal distribution."
    ":class:`XavierNormal <neupy.init.XavierNormal>`", "Glorot Xavier parameter initialization method based on the Normal distribution."
    ":class:`XavierUniform <neupy.init.XavierUniform>`", "Glorot Xavier parameter initialization method based on the Uniformal distribution."

.. _cheatsheet-error-function:

Error functions
***************

.. csv-table::
    :header: "Function name", "Description"

    ":class:`mae <neupy.algorithms.gd.errors.mae>`", "Mean absolute error"
    ":class:`mse <neupy.algorithms.gd.errors.mse>`", "Mean squared error"
    ":class:`rmse <neupy.algorithms.gd.errors.rmse>`", "Root mean squared error"
    ":class:`msle <neupy.algorithms.gd.errors.msle>`", "Mean squared logarithmic error"
    ":class:`rmsle <neupy.algorithms.gd.errors.rmsle>`", "Root mean squared logarithmic error"
    ":class:`binary_crossentropy <neupy.algorithms.gd.errors.binary_crossentropy>`", "Cross entropy error function for the binary classification"
    ":class:`categorical_crossentropy <neupy.algorithms.gd.errors.categorical_crossentropy>`", "Cross entropy error function for the multi-class classification"
    ":class:`binary_hinge <neupy.algorithms.gd.errors.binary_hinge>`", "Hinge error function for the binary classification"
    ":class:`categorical_hinge <neupy.algorithms.gd.errors.categorical_hinge>`", "Hinge error function for the multi-class classification"


Datasets
********

.. csv-table::
    :header: "Dataset name", "Description"

    ":class:`load_digits <neupy.datasets.digits.load_digits>`", "Load 10 discrete digit images with shape (6, 4)"
    ":class:`make_digits <neupy.datasets.digits.make_digits>`", "Load discrete digits that has additional noise."
    ":class:`make_reber <neupy.datasets.reber.make_reber>`", "Generate list of words valid by Grammar rules."
    ":class:`make_reber_classification <neupy.datasets.reber.make_reber_classification>`", "Generate random dataset for Reber grammar classification."
