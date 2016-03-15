NeuPy v0.2.0
============

NeuPy is a Python library for Artificial Neural Networks.
You can run and test different Neural Network algorithms.

|Travis|_

.. |Travis| image:: https://api.travis-ci.org/itdxer/neupy.png?branch=master
.. _Travis: https://travis-ci.org/itdxer/neupy

.. |Coveralls| image:: https://coveralls.io/repos/github/itdxer/neupy/badge.svg?branch=master
.. _Coveralls: https://coveralls.io/github/itdxer/neupy?branch=master

Installation
------------

.. code-block:: bash

    $ pip install neupy

Links
-----

* `Documentation <http://neupy.com>`_
* `Issues <https://github.com/itdxer/neupy/issues>`_
* `Tutorials <http://neupy.com/archive.html>`_
* `Available algorithms <http://neupy.com/docs/algorithms.html>`_

Tutorials
---------

* `Password recovery <http://neupy.com/2015/09/21/password_recovery.html>`_
* `Discrete Hopfield Network <http://neupy.com/2015/09/20/discrete_hopfield_network.html>`_
* `Boston house-prices dataset <http://neupy.com/2015/07/04/boston_house_prices_dataset.html>`_
* `Visualize Backpropagation Algorithms <http://neupy.com/2015/07/04/visualize_backpropagation_algorithms.html>`_

Examples
--------

Gradient based Neural Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* `MNIST, Multilayer perceptron <examples/gd/mnist_mlp.py>`_
* `Rectangle images, Multilayer perceptron <examples/gd/rectangles_mlp.py>`_
* `MNIST, Denoising Autoencoder <examples/gd/mnist_autoencoder.py>`_
* `Boston House Price prediction, Hessian algorithm <examples/gd/boston_price_prediction.py>`_
* `Learning Algorithms Visualization, Gradient Descent, Momentum, RPROP and Conjugate Gradient <examples/gd/gd_algorithms_visualization.py>`_

Competitive Neural Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* `Simple SOFM example <examples/competitive/sofm_basic.py>`_

Neural Networks with Radial Basis Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* `Classify iris dataset, Probabilistic Neural Network (PNN) <examples/rbfn/pnn_iris.py>`_
* `Regression using Diabetes dataset, Generilized Neural Nerwork (GRNN) <examples/rbfn/grnn_params_selection.py>`_
* `Music-Speech audio classification, Probabilistic Neural Network (PNN) <examples/rbfn/music_speech>`_

Memory based Neural Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* `Sinus function approximation, CMAC <examples/memory/cmac_basic.py>`_
* `Visualize Discrete Hopfield Neural Network energy function <examples/memory/dhn_energy_func.py>`_
* `Password recovery, Discrete Hopfield Neural Network <examples/memory/password_recovery.py>`_

Dependence
----------

* Python 2.7, 3.3, 3.4
* Theano >= 0.7.0
* NumPy >= 1.9.0
* SciPy >= 0.14.0
* Matplotlib >= 1.4.0

Next steps
----------

* Adding convolutional neural network layers (https://github.com/itdxer/neupy/issues/56)
* Adding reccurent neural network layers (https://github.com/itdxer/neupy/issues/57)
* Bug fixing and version stabilization (https://github.com/itdxer/neupy/issues?q=is%3Aissue+is%3Aopen+label%3Abug)
* Speeding up algorithms
* Adding more algorithms

Library includes
----------------

* Radial Basis Functions Networks (RBFN)
* GradientDescent and different optimization for it
* Neural Network Ensembles
* Associative and Autoasociative Memory
* Competitive Networks
* Step update algorithms for backpropagation
* Weight control algorithms for backpropagation
* Basic Linear Networks

Algorithms
----------

* GradientDescent

  * Classic Gradient Descent
  * Mini-batch Gradient Descent
  * Conjugate Gradient

    * Fletcher-Reeves
    * Polak-Ribiere
    * Hestenes-Stiefel
    * Conjugate Descent
    * Liu-Storey
    * Dai-Yuan

  * quasi-Newton with Wolfe search

    * BFGS
    * DFP
    * PSB
    * SR1

  * Levenberg-Marquardt
  * Hessian (Newton's method)
  * Hessian diagonal
  * Momentum
  * Nesterov Momentum
  * RPROP
  * iRPROP+
  * QuickProp
  * Adadelta
  * Adagrad
  * RMSProp
  * Adam
  * AdaMax

* Weight update rules

  * Weight Decay
  * Weight Elimination

* Learning rate update rules

  * Leak step adaptation
  * Error difference Update
  * Linear search using Golden Search or Brent
  * Search than converge
  * Simple Step Minimization

* Ensembles

  * Mixture of Experts
  * Dynamically Averaged Network (DAN)

* Radial Basis Functions Networks (RBFN)

  * Generalized Regression Neural Network (GRNN)
  * Probabilistic Neural Network (PNN)
  * Radial basis function K-means

* Autoasociative Memory

  * Discrete BAM Network
  * CMAC Network
  * Discrete Hopfield Network

* Competitive Networks

  * Adaptive Resonance Theory (ART1) Network
  * Self-Organizing Feature Map (SOFM or SOM)

* Linear networks

  * Perceptron
  * LMS Network
  * Modified Relaxation Network

* Associative

  * OJA
  * Kohonen
  * Instar
  * Hebb

Tests
-----

.. code-block:: bash

    $ pip install tox
    $ tox
