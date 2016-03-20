Home
====

NeuPy is a Python library for Artificial Neural Networks.
You can run and test different Neural Network algorithms.

.. image:: ../_static/img/mnist-solution-code.png
    :width: 100%
    :align: center

Algorithms
----------

* Algorithms that use Backpropagation training approach

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

* Algorithms that penalize weights

  * Weight Decay
  * Weight Elimination

* Algorithms that update learning rate

  * Adaptive Learning Rate
  * Error difference Update
  * Linear search using Golden Search or Brent
  * Search than converge
  * Simple Step Minimization

* Ensembles

  * Mixture of Experts
  * Dynamically Averaged Network (DAN)

* Neural Networks with Radial Basis Functions (RBFN)

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

Next steps
----------

* Adding convolutional neural network layers (`Issue #56 <https://github.com/itdxer/neupy/issues/56>`_)
* Adding reccurent neural network layers (`Issue #57 <https://github.com/itdxer/neupy/issues/57>`_)
* Bug fixing and version stabilization  (`Known bugs <https://github.com/itdxer/neupy/issues?q=is%3Aissue+is%3Aopen+label%3Abug>`_)
* Speeding up algorithms
* Adding more algorithms

Dependencies
------------

* Python 2.7, 3.3, 3.4
* Theano >= 0.7.0
* NumPy >= 1.9.0
* SciPy >= 0.14.0
* Matplotlib >= 1.4.0

Tests
-----

.. code-block:: bash

    $ pip install tox
    $ tox
