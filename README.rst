|Travis|_

.. |Travis| image:: https://api.travis-ci.org/itdxer/neupy.png?branch=master
.. _Travis: https://travis-ci.org/itdxer/neupy

.. |Coveralls| image:: https://coveralls.io/repos/github/itdxer/neupy/badge.svg?branch=master
.. _Coveralls: https://coveralls.io/github/itdxer/neupy?branch=master

NeuPy v0.2.3 (beta)
===================

NeuPy is a Python library for Artificial Neural Networks.

.. image:: site/_static/img/mnist-solution-code.png
    :width: 80%
    :align: center

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

MLP Neural Networks
~~~~~~~~~~~~~~~~~~~

* `MNIST, Multilayer perceptron <examples/gd/mnist_mlp.py>`_
* `Rectangle images, Multilayer perceptron <examples/gd/rectangles_mlp.py>`_
* `MNIST, Denoising Autoencoder <examples/gd/mnist_autoencoder.py>`_
* `Boston House Price prediction, Hessian algorithm <examples/gd/boston_price_prediction.py>`_
* `Learning Algorithms Visualization, Gradient Descent, Momentum, RPROP and Conjugate Gradient <examples/gd/gd_algorithms_visualization.py>`_
* `IMDB review classification using CBOW and RPROP MLP <examples/gd/word_embedding>`_

Convolutional Neural Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* `MNIST CNN <examples/gd/mnist_cnn.py>`_
* `CIFAR10 CNN <examples/gd/cifar10_cnn.py>`_

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

Dependencies
------------

* Python 2.7, 3.4
* Theano >= 0.8.1
* NumPy >= 1.9.0
* SciPy >= 0.14.0
* Matplotlib >= 1.4.0

Next steps
----------

* Adding reccurent neural network layers (`Issue #57 <https://github.com/itdxer/neupy/issues/57>`_)
* Bug fixing and version stabilization  (`Known bugs <https://github.com/itdxer/neupy/issues?q=is%3Aissue+is%3Aopen+label%3Abug>`_)
* Speeding up algorithms
* Adding more algorithms

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
