Classes and functions
=====================

.. contents::

Algorithms
**********

Algorithms that use Backpropagation training approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Weight update rules
~~~~~~~~~~~~~~~~~~~

.. csv-table::
    :header: "Class name", "Name"

    :network:`WeightDecay`, Weight Decay
    :network:`WeightElimination`, Weight Elimination

Learning rate update rules
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
    :header: "Class name", "Name"

    :network:`LeakStepAdaptation`, Leak Step Adaptation
    :network:`ErrDiffStepUpdate`, Error difference Update
    :network:`LinearSearch`, Linear search by Golden Search or Brent
    :network:`SearchThenConverge`, Search than converge
    :network:`SimpleStepMinimization`, Simple Step Minimization

Ensembles
~~~~~~~~~

.. csv-table::
    :header: "Class name", "Name"

    :network:`MixtureOfExperts`, Mixture of Experts
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

Linear networks
~~~~~~~~~~~~~~~~

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
    :network:`HebbRule`, Hebb

Layers
******

Input and hidden layers
~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
    :header: "Class name", "Description"

    ":layer:`Linear`", "Layer with linear activation function."
    ":layer:`Sigmoid`", "Layer with sigmoid activation function."
    ":layer:`HardSigmoid`", "Layer with hard sigmoid activation function."
    ":layer:`Step`", "Layer with step activation function."
    ":layer:`Tanh`", "Layer with tanh activation function."
    ":layer:`Relu`", "Layer with rectifier activation function."
    ":layer:`Softplus`", "Layer with softplus activation function."
    ":layer:`Softmax`", "Layer with softmax activation function."
    ":layer:`Dropout`", "Dropout layer"

Output layers
~~~~~~~~~~~~~

.. csv-table::
    :header: "Class name", "Description"

    ":layer:`Output`", "Simple output layer which does not make any transformations"
    ":layer:`CompetitiveOutput`", "Competitive layer output"
    ":layer:`StepOutput`", "The behaviour for this output layer is the same as for step function."
    ":layer:`RoundedOutput`", "Layer round output value."
    ":layer:`ArgmaxOutput`", "Return number of feature that have maximum value for each sample."


Error functions
***************

.. csv-table::
    :header: "Function name", "Description"

    "mae", "Mean absolute error"
    "mse", "Mean squared error"
    "rmse", "Root mean squared error"
    "msle", "Mean squared logarithmic error"
    "rmsle", "Root mean squared logarithmic error"
    "categorical_crossentropy", "Cross entropy error"
    "binary_crossentropy", "Cross entropy error"
