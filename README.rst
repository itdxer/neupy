NeuPy
=====

NeuPy is a Python library for Artificial Neural Networks.
You can run and test different Neural Network algorithms.

|Travis|_

.. |Travis| image:: https://api.travis-ci.org/itdxer/neupy.png?branch=master
.. _Travis: https://travis-ci.org/itdxer/neupy

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

Dependence
----------

* Python 2.7, 3.3, 3.4
* NumPy >= 1.9.0
* SciPy >= 0.14.0
* Matplotlib >= 1.4.0

Next steps
----------

* Bug fixing and version stabilization
* Speeding up algorithms
* Adding more algorithms

Library support
---------------

* Radial Basis Functions Networks (RBFN)
* Backpropagation and different optimization for it
* Neural Network Ensembles
* Associative and Autoasociative Memory
* Competitive Networks
* Step update algorithms for backpropagation
* Weight control algorithms for backpropagation
* Basic Linear Networks

Algorithms
----------

* Backpropagation

  * Classic Gradient Descent
  * Mini-batch Gradient Descent
  * Conjugate Gradient

    * Fletcher-Reeves
    * Polak-Ribiere
    * Hestenes-Stiefel
    * Conjugate Descent
    * Liu-Storey
    * Dai-Yuan

  * quasi-Newton

    * BFGS
    * DFP
    * PSB
    * SR1

  * Levenberg-Marquardt
  * Hessian diagonal
  * Momentum
  * RPROP
  * iRPROP+
  * QuickProp

* Weight update rules

  * Weight Decay
  * Weight Elimination

* Learning rate update rules

  * Adaptive Learning Rate
  * Error difference Update
  * Linear search by Golden Search or Brent
  * Wolfe line search
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
