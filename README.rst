|Travis|_ |Coverage|_ |Dependency Status|_ |License|_

.. |Travis| image:: https://api.travis-ci.org/itdxer/neupy.png?branch=master
.. _Travis: https://travis-ci.org/itdxer/neupy

.. |Dependency Status| image:: https://dependencyci.com/github/itdxer/neupy/badge
.. _Dependency Status: https://dependencyci.com/github/itdxer/neupy

.. |License| image:: https://img.shields.io/badge/license-MIT-blue.svg
.. _License: https://github.com/itdxer/neupy/blob/master/LICENSE

.. |Coverage| image:: https://codecov.io/gh/itdxer/neupy/branch/master/graph/badge.svg
.. _Coverage: https://codecov.io/gh/itdxer/neupy


.. raw:: html

    <div align="center">
        <a href="http://neupy.com/2017/12/17/sofm_text_style.html">
        <img width="80%" src="https://github.com/itdxer/neupy/raw/master/site/neupy-logo.png">
        </a>
    </div>


NeuPy v0.8.2
============

NeuPy is a python library for prototyping and building neural networks. NeuPy uses Tensorflow as a computational backend for deep learning models.

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
* `Open Issues <https://github.com/itdxer/neupy/issues>`_ and ask questions.

Articles and Notebooks
----------------------

.. raw:: html

    <table border="0">
        <tr>
            <td border="0" width="30%">
                <a href="http://neupy.com/2018/03/26/making_art_with_growing_neural_gas.html#id1">
                <img src="site/_static/intro/gng-animation-intro.gif">
                </a>
            </td>
            <td border="0" valign="top">
                <h3><a href="http://neupy.com/2018/03/26/making_art_with_growing_neural_gas.html#id1">Growing Neural Gas</a></h3>
                <p>Growing Neural Gas is an algorithm that learns topological structure of the data.</p>
                <p>Code that generates animation can be found in <a href="https://github.com/itdxer/neupy/blob/master/notebooks/growing-neural-gas/Growing%20Neural%20Gas%20animated.ipynb">this ipython notebook</a></p>
            </td>
        </tr>
        <tr>
            <td border="0" width="30%">
                <a href="http://neupy.com/2018/03/26/making_art_with_growing_neural_gas.html">
                <img src="site/_static/intro/gng-art-intro.png">
                </a>
            </td>
            <td border="0" valign="top">
                <h3><a href="http://neupy.com/2018/03/26/making_art_with_growing_neural_gas.html">Making Art with Growing Neural Gas</a></h3>
                <p>Growing Neural Gas is another example of the algorithm that follows simple set of rules that on a large scale can generate complex patterns.</p>
                <p>Image on the left is a great example of the art style that can be generated with simple set fo rules.</p>
                <p>The main notebook that generates image can be found <a href="https://github.com/itdxer/neupy/blob/master/notebooks/growing-neural-gas/Making%20Art%20with%20Growing%20Neural%20Gas.ipynb">here</a></p>
            </td>
        </tr>
        <tr>
            <td border="0" width="30%">
                <a href="http://neupy.com/2017/12/09/sofm_applications.html">
                <img src="site/_static/intro/sofm-dnn-intro.png">
                </a>
            </td>
            <td border="0" valign="top">
                <h3><a href="http://neupy.com/2017/12/09/sofm_applications.html">Self-Organizing Maps and Applications</a></h3>
                <p>
                    Self-Organazing Maps (SOM or SOFM) is a very simple and powerful algorithm that has a wide variety of applications. This articles covers some of them, including:

                    <ul>
                        <li>Visualizing Convolutional Neural Networks</li>
                        <li>Data topology learning</li>
                        <li>High-dimensional data visualization</li>
                        <li>Clustering</li>
                    </ul>
                </p>
            </td>
        </tr>
        <tr>
            <td border="0" width="30%">
                <a href="https://github.com/itdxer/neupy/blob/master/notebooks/Visualizing%20CNN%20based%20on%20Pre-trained%20VGG19.ipynb">
                <img src="site/_static/intro/cnn-vis-intro.png">
                </a>
            </td>
            <td border="0" valign="top">
                <h3><a href="https://github.com/itdxer/neupy/blob/master/notebooks/Visualizing%20CNN%20based%20on%20Pre-trained%20VGG19.ipynb">Visualizing CNN based on Pre-trained VGG19</a></h3>
                <p>This notebook shows how you can easely explore reasons behind convolutional network predictions and understand what type of features has been learned in different layers of the network.</p>
                <p>In addition, this notebook shows how to use neural network architectures in NeuPy, like VGG19, with pre-trained parameters.</p>
            </td>
        </tr>
        <tr>
            <td border="0" width="30%">
                <a href="http://neupy.com/2015/07/04/visualize_backpropagation_algorithms.html">
                <img src="site/_static/intro/vis-gd-intro.png">
                </a>
            </td>
            <td border="0" valign="top">
                <h3><a href="http://neupy.com/2015/07/04/visualize_backpropagation_algorithms.html">Visualize Algorithms based on the Backpropagation</a></h3>
                <p>Image on the left shows comparison between paths that different algorithm take along the descent path. It's interesting to see how much information about the algorithm can be extracted from simple trajectory paths. All of this covered and explained in the article.</p>
            </td>
        </tr>
        <tr>
            <td border="0" width="30%">
                <a href="http://neupy.com/2016/12/17/hyperparameter_optimization_for_neural_networks.html">
                <img src="site/_static/intro/hyperopt-intro.png">
                <img src="site/_static/intro/hyperopt-2-intro.png">
                </a>
            </td>
            <td border="0" valign="top">
                <h3><a href="http://neupy.com/2016/12/17/hyperparameter_optimization_for_neural_networks.html">Hyperparameter optimization for Neural Networks</a></h3>
                <p>
                    This article covers different approaches for hyperparameter optimization.
                    <ul>
                    <li>Grid Search</li>
                    <li>Random Search</li>
                    <li>Hand-tuning</li>
                    <li>Gaussian Process with Expected Improvement</li>
                    <li>Tree-structured Parzen Estimators (TPE)</li>
                    </ul>
                </p>
            </td>
        </tr>
        <tr>
            <td border="0" width="30%">
                <a href="http://neupy.com/2017/12/13/sofm_art.html">
                <img src="site/_static/intro/sofm-art-intro.png">
                </a>
            </td>
            <td border="0" valign="top">
                <h3><a href="http://neupy.com/2017/12/13/sofm_art.html">The Art of SOFM</a></h3>
                <p>In this article, I just want to show how beautiful sometimes can be a neural network. I think, it’s quite rare that algorithm can not only extract knowledge from the data, but also produce something beautiful using exactly the same set of training rules without any modifications.</p>
            </td>
        </tr>
        <tr>
            <td border="0" width="30%">
                <a href="http://neupy.com/2015/09/20/discrete_hopfield_network.html">
                <img src="site/_static/intro/discrete-hn-intro.png">
                </a>
            </td>
            <td border="0" valign="top">
                <h3><a href="http://neupy.com/2015/09/20/discrete_hopfield_network.html">Discrete Hopfield Network</a></h3>
                <p>Article with extensive theoretical background about Discrete Hopfield Network. It also has example that show advantages and limitations of the algorithm.</p>
                <p>Image on the left is a visulatization of the information stored in the network. This picture not only visualizes network's memory, ot shows everything network knows about the world.</p>
            </td>
        </tr>
        <tr>
            <td border="0" width="30%">
                <a href="http://neupy.com/2017/12/17/sofm_text_style.html">
                <img src="site/_static/intro/sofm-neupy-logo-intro.png">
                </a>
            </td>
            <td border="0" valign="top">
                <h3><a href="http://neupy.com/2017/12/17/sofm_text_style.html">Create unique text-style with SOFM</a></h3>
                <p>This article describes step-by-step solution that allow to generate unique styles with arbitrary text.</p>
            </td>
        </tr>
        <tr>
            <td border="0" width="30%">
                <a href="https://github.com/itdxer/neupy/blob/master/notebooks/Playing%20with%20MLP%20visualizations.ipynb">
                <img src="site/_static/intro/mlp-vis-intro.png">
                </a>
            </td>
            <td border="0" valign="top">
                <h3><a href="https://github.com/itdxer/neupy/blob/master/notebooks/Playing%20with%20MLP%20visualizations.ipynb">Playing with MLP visualizations</a></h3>
                <p>This notebook shows interesting ways to look inside of your MLP network.</p>
            </td>
        </tr>
        <tr>
            <td border="0" width="30%">
                <a href="https://github.com/itdxer/neupy/tree/master/examples/reinforcement_learning/vin">
                <img src="site/_static/intro/vin-intro.png">
                </a>
            </td>
            <td border="0" valign="top">
                <h3><a href="https://github.com/itdxer/neupy/tree/master/examples/reinforcement_learning/vin">Exploring world with Value Iteration Network (VIN)</a></h3>
                <p>One of the basic applications of the Value Iteration Network that learns how to find an optimal path between two points in the environment with obstacles.</p>
            </td>
        </tr>
        <tr>
            <td border="0" width="30%">
                <a href="https://github.com/itdxer/neupy/tree/master/examples/boltzmann_machine">
                <img src="site/_static/intro/boltzman-machine-intro.png">
                </a>
            </td>
            <td border="0" valign="top">
                <h3><a href="https://github.com/itdxer/neupy/tree/master/examples/boltzmann_machine">Features learned by Restricted Boltzmann Machine (RBM)</a></h3>
                <p>Set of examples that use and explore knowledge extracted by Restricted Boltzmann Machine</p>
            </td>
        </tr>
    </table>
