|Travis|_ |Dependency Status|_

.. |Travis| image:: https://api.travis-ci.org/itdxer/neupy.png?branch=master
.. _Travis: https://travis-ci.org/itdxer/neupy

.. |Dependency Status| image:: https://dependencyci.com/github/itdxer/neupy/badge
.. _Dependency Status: https://dependencyci.com/github/itdxer/neupy

NeuPy v0.3.0 (beta)
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
* `Articles <http://neupy.com/archive.html>`_
* `Available algorithms <http://neupy.com/docs/cheatsheet.html#algorithms>`_

Articles
--------

* `Password recovery <http://neupy.com/2015/09/21/password_recovery.html>`_
* `Discrete Hopfield Network <http://neupy.com/2015/09/20/discrete_hopfield_network.html>`_
* `Boston house-prices dataset <http://neupy.com/2015/07/04/boston_house_prices_dataset.html>`_
* `Visualize Backpropagation Algorithms <http://neupy.com/2015/07/04/visualize_backpropagation_algorithms.html>`_
* `MNIST classification <http://neupy.com/docs/quickstart.html#mnist-classification>`_

Examples
--------

MLP Neural Networks
~~~~~~~~~~~~~~~~~~~

* `MNIST, Multilayer perceptron <examples/mlp/mnist_mlp.py>`_
* `Rectangle images, Multilayer perceptron <examples/mlp/rectangles_mlp.py>`_
* `Boston House Price prediction, Hessian algorithm <examples/mlp/boston_price_prediction.py>`_
* `Learning Algorithms Visualization, Gradient Descent, Momentum, RPROP and Conjugate Gradient <examples/mlp/gd_algorithms_visualization.py>`_
* `IMDB review classification using CBOW and RPROP MLP <examples/mlp/imdb_review_classification>`_

Convolutional Neural Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* `MNIST CNN <examples/cnn/mnist_cnn.py>`_
* `CIFAR10 CNN <examples/cnn/cifar10_cnn.py>`_

Autoencoders
~~~~~~~~~~~~

* `MNIST, Denoising Autoencoder <examples/autoencoder/denoising_autoencoder.py>`_
* `MNIST, Convolutional Autoencoder <examples/autoencoder/conv_autoencoder.py>`_
* `MNIST, Stacked Convolutional Autoencoders <examples/autoencoder/stacked_conv_autoencoders.py>`_

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

* Python 2.7, 3.4, 3.5
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
