|Travis|_ |Dependency Status|_ |License|_

.. |Travis| image:: https://api.travis-ci.org/itdxer/neupy.png?branch=master
.. _Travis: https://travis-ci.org/itdxer/neupy

.. |Dependency Status| image:: https://dependencyci.com/github/itdxer/neupy/badge
.. _Dependency Status: https://dependencyci.com/github/itdxer/neupy

.. |License| image:: https://img.shields.io/badge/license-MIT-blue.svg
.. _License: https://github.com/itdxer/neupy/blob/master/LICENSE


NeuPy v0.6.0
============

NeuPy is a Python library for Artificial Neural Networks. NeuPy supports many different types of Neural Networks from a simple perceptron to deep learning models.

.. image:: https://github.com/itdxer/neupy/raw/master/site/_static/img/mnist-solution-code.png
    :width: 80%
    :align: center

Installation
------------

.. code-block:: bash

    $ pip install neupy

User Guide
----------

* `Install NeuPy <http://neupy.com/pages/installation.html>`_
* Check the `tutorials <http://neupy.com/docs/tutorials.html>`_
* Learn more about NeuPy in the `documentation <http://neupy.com/pages/documentation.html>`_
* Explore lots of different `neural network algorithms <http://neupy.com/pages/cheatsheet.html>`_.
* Read `articles <http://neupy.com/archive.html>`_ and learn more about Neural Networks.

Links
-----

* `Tutorials <http://neupy.com/docs/tutorials.html>`_
* `Documentation <http://neupy.com/pages/documentation.html>`_
* `Articles <http://neupy.com/archive.html>`_
* `Cheat sheet <http://neupy.com/pages/cheatsheet.html>`_
* `Open Issues <https://github.com/itdxer/neupy/issues>`_

Articles
--------

* `Hyperparameter optimization for Neural Networks  <http://neupy.com/2016/12/17/hyperparameter_optimization_for_neural_networks.html>`_
* `Visualize Backpropagation Algorithms <http://neupy.com/2015/07/04/visualize_backpropagation_algorithms.html>`_
* `MNIST classification <http://neupy.com/2016/11/12/mnist_classification.html>`_
* `Predict prices for houses in the area of Boston <http://neupy.com/2015/07/04/boston_house_prices_dataset.html>`_
* `Password recovery <http://neupy.com/2015/09/21/password_recovery.html>`_
* `Discrete Hopfield Network <http://neupy.com/2015/09/20/discrete_hopfield_network.html>`_

Jupyter Notebooks
-----------------

* `Hyperparameter optimization for Neural Networks <https://github.com/itdxer/neupy/blob/master/notebooks/Hyperparameter%20optimization%20for%20Neural%20Networks.ipynb>`_
* `Playing with MLP visualizations <https://github.com/itdxer/neupy/blob/master/notebooks/Playing%20with%20MLP%20visualizations.ipynb>`_
* `Visualizing CNN based on Pre-trained VGG19 <https://github.com/itdxer/neupy/blob/master/notebooks/Visualizing%20CNN%20based%20on%20Pre-trained%20VGG19.ipynb>`_
* `Looking inside of the VGG19 using SOFM  <https://github.com/itdxer/neupy/blob/master/notebooks/Looking%20inside%20of%20the%20VGG19%20using%20SOFM.ipynb>`_

Examples
--------

Convolutional Neural Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Image classification
++++++++++++++++++++

* `MNIST CNN <https://github.com/itdxer/neupy/tree/master/examples/cnn/mnist_cnn.py>`_
* `CIFAR10 CNN <https://github.com/itdxer/neupy/tree/master/examples/cnn/cifar10_cnn.py>`_
* `Pre-trained AlexNet CNN <https://github.com/itdxer/neupy/tree/master/examples/cnn/alexnet.py>`_
* `Pre-trained VGG16 CNN <https://github.com/itdxer/neupy/tree/master/examples/cnn/vgg16.py>`_
* `Pre-trained VGG19 CNN <https://github.com/itdxer/neupy/tree/master/examples/cnn/vgg19.py>`_
* `Pre-trained SqueezeNet <https://github.com/itdxer/neupy/tree/master/examples/cnn/squeezenet.py>`_
* `GoogleNet <https://github.com/itdxer/neupy/tree/master/examples/cnn/googlenet.py>`_
* `Inception v3 <https://github.com/itdxer/neupy/tree/master/examples/cnn/inception_v3.py>`_
* `ResNet 50 <https://github.com/itdxer/neupy/tree/master/examples/cnn/resnet50.py>`_

Reinforcement Learning
~~~~~~~~~~~~~~~~~~~~~~

* `Neural Network plays CartPole game <https://github.com/itdxer/neupy/tree/master/examples/reinforcement_learning/rl_cartpole.py>`_
* `Value Iteration Network (VIN) <https://github.com/itdxer/neupy/tree/master/examples/reinforcement_learning/vin>`_

Recurrent Neural Networks
~~~~~~~~~~~~~~~~~~~~~~~~~

* `Reber Grammar Classification, sequence input and fixed-size output with GRU <https://github.com/itdxer/neupy/tree/master/examples/rnn/reber_gru.py>`_
* `Generate Shakespeare text, sequence input and sequence output with LSTM <https://github.com/itdxer/neupy/tree/master/examples/rnn/shakespeare_lstm.py>`_

Autoencoders
~~~~~~~~~~~~

* `MNIST, Denoising Autoencoder <https://github.com/itdxer/neupy/tree/master/examples/autoencoder/denoising_autoencoder.py>`_
* `MNIST, Convolutional Autoencoder <https://github.com/itdxer/neupy/tree/master/examples/autoencoder/conv_autoencoder.py>`_
* `MNIST, Stacked Convolutional Autoencoders <https://github.com/itdxer/neupy/tree/master/examples/autoencoder/stacked_conv_autoencoders.py>`_
* `MNIST, Variational Autoencoder <https://github.com/itdxer/neupy/tree/master/examples/autoencoder/variational_autoencoder.py>`_

Boltzmann Machine
~~~~~~~~~~~~~~~~~

* `Feature Learning from the MNIST Images, Restricted Boltzmann Machine (RBM) <https://github.com/itdxer/neupy/tree/master/examples/boltzmann_machine/rbm_mnist.py>`_
* `Gibbs sampling using face images, Restricted Boltzmann Machine (RBM) <https://github.com/itdxer/neupy/tree/master/examples/boltzmann_machine/rbm_faces_sampling.py>`_

MLP Neural Networks
~~~~~~~~~~~~~~~~~~~

Regression
++++++++++

* `Boston House Price prediction, Hessian algorithm <https://github.com/itdxer/neupy/tree/master/examples/mlp/boston_price_prediction.py>`_

Image classification
++++++++++++++++++++

* `MNIST, Multilayer perceptron <https://github.com/itdxer/neupy/tree/master/examples/mlp/mnist_mlp.py>`_
* `Rectangle images, Multilayer perceptron <https://github.com/itdxer/neupy/tree/master/examples/mlp/rectangles_mlp.py>`_

Visualizations
++++++++++++++

* `Learning Algorithms Visualization, Gradient Descent, Momentum, RPROP and Conjugate Gradient <https://github.com/itdxer/neupy/tree/master/examples/mlp/gd_algorithms_visualization.py>`_

Classification
++++++++++++++

* `IMDB review classification using CBOW and RPROP MLP <https://github.com/itdxer/neupy/tree/master/examples/mlp/imdb_review_classification>`_
* `MLP with categorical and numerical features <https://github.com/itdxer/neupy/tree/master/examples/mlp/mix_categorical_numerical_inputs.py>`_

Competitive Neural Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~

SOFM Clustering
+++++++++++++++

* `Simple SOFM example <https://github.com/itdxer/neupy/tree/master/examples/competitive/sofm_basic.py>`_
* `Clustering iris dataset using SOFM <https://github.com/itdxer/neupy/tree/master/examples/competitive/sofm_iris_clustering.py>`_

SOFM Data topology learning
+++++++++++++++++++++++++++

* `Learning half-circle topology with SOFM <https://github.com/itdxer/neupy/tree/master/examples/competitive/sofm_moon_topology.py>`_

Explore SOFM features
+++++++++++++++++++++

* `Compare feature grid types for SOFM <https://github.com/itdxer/neupy/tree/master/examples/competitive/sofm_compare_grid_types.py>`_
* `Compare weight initialization methods for SOFM <https://github.com/itdxer/neupy/tree/master/examples/competitive/sofm_compare_weight_init.py>`_

SOFM Visualizations
+++++++++++++++++++

* `Visualize digit images in 2D space with SOFM <https://github.com/itdxer/neupy/tree/master/examples/competitive/sofm_digits.py>`_
* `Embedding 30-dimensional dataset into 2D and building heatmap visualization for SOFM <https://github.com/itdxer/neupy/tree/master/examples/competitive/sofm_heatmap_visualization.py>`_

LVQ
+++

* `Reduce number of training samples in iris dataset with LVQ3 <https://github.com/itdxer/neupy/tree/master/examples/competitive/reduce_iris_sample_size_lvq.py>`_

Neural Networks with Radial Basis Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* `Classify iris dataset, Probabilistic Neural Network (PNN) <https://github.com/itdxer/neupy/tree/master/examples/rbfn/pnn_iris.py>`_
* `Regression using Diabetes dataset, Generilized Neural Nerwork (GRNN) <https://github.com/itdxer/neupy/tree/master/examples/rbfn/grnn_params_selection.py>`_
* `Music-Speech audio classification, Probabilistic Neural Network (PNN) <https://github.com/itdxer/neupy/tree/master/examples/rbfn/music_speech>`_

Memory based Neural Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* `Sinus function approximation, CMAC <https://github.com/itdxer/neupy/tree/master/examples/memory/cmac_basic.py>`_
* `Visualize Discrete Hopfield Neural Network energy function <https://github.com/itdxer/neupy/tree/master/examples/memory/dhn_energy_func.py>`_
* `Password recovery, Discrete Hopfield Neural Network <https://github.com/itdxer/neupy/tree/master/examples/memory/password_recovery.py>`_

Dependencies
------------

* Python 2.7, 3.4, 3.5, 3.6
* Theano == 0.9.0
* NumPy >= 1.9.0
* SciPy >= 0.19.0
* Matplotlib >= 1.4.0
* graphviz == 0.5.1
