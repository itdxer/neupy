.. _tutorials:

Tutorials
=========

Tutorial Articles
-----------------

There are a few articles that can help you to start working with NeuPy. They provide a solution to different problems and explain each step of the overall process.

* :ref:`mnist-classification`
* :ref:`boston-house-price`

Code Examples
-------------

NeuPy is very intuitive and it's easy to read and understand the code. To learn more about different Neural Network types you can check these code examples.

.. raw:: html

    <table border="1" style="background-color: white;">
        <tr>
          <td colspan="2" style="text-align: center;">
              <h3>Deep Learning</h3>
          </td>
        </tr>
        <tr>
            <td width="35%" valign="top" style="text-align: center; padding: 10px;">
                <h3>Image classification - CNN</h3>
                <img width="100%" src="https://raw.githubusercontent.com/itdxer/neupy/master/site/_static/intro/sofm-dnn-intro.png">
                <img width="100%" style="padding-top: 20px;" src="https://raw.githubusercontent.com/itdxer/neupy/master/site/_static/intro/cnn-vis-intro.png">
            </td>
            <td valign="top" style="padding: 10px;">
                <div><b>Model training:</b></div>
                <ul>
                  <li><a href="https://github.com/itdxer/neupy/tree/master/examples/cnn/mnist_cnn.py">MNIST</a></li>
                  <li><a href="https://github.com/itdxer/neupy/tree/master/examples/cnn/cifar10_cnn.py">CIFAR 10</a></li>
                  <li><a href="https://github.com/itdxer/neupy/tree/master/examples/autoencoder/stacked_conv_autoencoders.py">MNIST semi-supervised training with stacked autoencoders</a></li>
                </ul>

                <div><b>Pre-trained models:</b></div>
                <ul>
                  <li><a href="https://github.com/itdxer/neupy/tree/master/examples/cnn/vgg16.py">VGG16</a></li>
                  <li><a href="https://github.com/itdxer/neupy/tree/master/examples/cnn/vgg19.py">VGG19</a></li>
                  <li><a href="https://github.com/itdxer/neupy/tree/master/examples/cnn/squeezenet.py">SqueezeNet</a></li>
                  <li><a href="https://github.com/itdxer/neupy/tree/master/examples/cnn/resnet50.py">ResNet 50</a></li>
                </ul>
                <div><b>Architectures:</b></div>
                <ul>
                  <li><a href="https://github.com/itdxer/neupy/tree/master/examples/cnn/alexnet.py">AlexNet</a></li>
                  <li><a href="https://github.com/itdxer/neupy/tree/master/examples/cnn/googlenet.py">GoogleNet</a></li>
                  <li><a href="https://github.com/itdxer/neupy/tree/master/examples/cnn/inception_v3.py">Inception v3</a></li>
                </ul>
                <div><b>Visualizations:</b></div>
                <ul>
                  <li><a href="https://github.com/itdxer/neupy/blob/master/notebooks/Visualizing%20CNN%20based%20on%20Pre-trained%20VGG19.ipynb">Visualizing CNN based on the pre-trained VGG19</a></li>
                  <li><a href="https://github.com/itdxer/neupy/blob/master/notebooks/Looking%20inside%20of%20the%20VGG19%20using%20SOFM.ipynb">Looking inside of the VGG19 using SOFM</a></li>
                </ul>
            </td>
        </tr>
        <tr>
            <td width="35%" valign="top" style="text-align: center; padding: 10px;">
                <h3>Multilayer Perceptron (MLP)</h3>
                <img width="100%" src="https://raw.githubusercontent.com/itdxer/neupy/master/site/_static/intro/vis-gd-intro.png">
            </td>
            <td valign="top" style="padding: 10px;">
              <div><b>Classification:</b></div>
              <ul>
                  <li><a href="https://github.com/itdxer/neupy/tree/master/examples/mlp/mnist_mlp.py">MNIST</a></li>
                  <li><a href="https://github.com/itdxer/neupy/tree/master/examples/mlp/imdb_review_classification">IMDB review classification using CBOW and RPROP MLP</a></li>
                  <li><a href="https://github.com/itdxer/neupy/tree/master/examples/mlp/mix_categorical_numerical_inputs.py">MLP with categorical and numerical features</a></li>
              </ul>
              <div><b>Regression:</b></div>
              <ul>
                  <li><a href="https://github.com/itdxer/neupy/tree/master/examples/mlp/boston_price_prediction.py">Boston house price prediction</a></li>
              </ul>
              <div><b>Visualizations:</b></div>
              <ul>
                  <li><a href="https://github.com/itdxer/neupy/tree/master/examples/mlp/gd_algorithms_visualization.py">Visualizing training process for different algorithms</a></li>
                  <li><a href="https://github.com/itdxer/neupy/blob/master/notebooks/Playing%20with%20MLP%20visualizations.ipynb">MLP Visualizations</a></li>
              </ul>
            </td>
        </tr>
        <tr>
            <td width="35%" valign="top" style="text-align: center; padding: 10px;">
                <h3>Recurrent Neural Networks (RNN)</h3>
            </td>
            <td valign="top" style="padding: 10px;">
              <ul>
                <li><a href="https://github.com/itdxer/neupy/tree/master/examples/rnn/reber_gru.py">Reber Grammar Classification, GRU</a></li>
                <li><a href="https://github.com/itdxer/neupy/tree/master/examples/rnn/shakespeare_lstm.py">Shakespear text generation, LSTM</a></li>
              </ul>
            </td>
        </tr>
        <tr>
            <td width="35%" valign="top" style="text-align: center; padding: 10px;">
                <h3>Autoencoders</h3>
            </td>
            <td valign="top" style="padding: 10px;">
              <ul>
                <li><a href="https://github.com/itdxer/neupy/tree/master/examples/autoencoder/denoising_autoencoder.py">MNIST, Denoising Autoencoder</a></li>
                <li><a href="https://github.com/itdxer/neupy/tree/master/examples/autoencoder/conv_autoencoder.py">MNIST, Convolutional Autoencoder</a></li>
                <li><a href="https://github.com/itdxer/neupy/tree/master/examples/autoencoder/stacked_conv_autoencoders.py">MNIST, Stacked Convolutional Autoencoders</a></li>
                <li><a href="https://github.com/itdxer/neupy/tree/master/examples/autoencoder/variational_autoencoder.py">MNIST, Variational Autoencoder</a></li>
              </ul>
            </td>
        </tr>
        <tr>
            <td width="35%" valign="top" style="text-align: center; padding: 10px;">
                <h3>Reinforcement Learning (RL)</h3>
                <img width="100%" src="https://raw.githubusercontent.com/itdxer/neupy/master/site/_static/intro/vin-intro.png" width="100%">
            </td>
            <td valign="top" style="padding: 10px;">
              <ul>
                <li><a href="https://github.com/itdxer/neupy/tree/master/examples/reinforcement_learning/rl_cartpole.py">Network plays CartPole game</a></li>
                <li><a href="https://github.com/itdxer/neupy/tree/master/examples/reinforcement_learning/vin">Value Iteration Networks (VIN)</a></li>
              </ul>
            </td>
        </tr>
        <tr>
            <td width="35%" valign="top" style="text-align: center; padding: 10px;">
                <h3>Restricted Boltzmann Machine (RBM)</h3>
                <img width="100%" src="https://raw.githubusercontent.com/itdxer/neupy/master/site/_static/intro/boltzman-machine-intro.png">
            </td>
            <td valign="top" style="padding: 10px;">
              <ul>
                <li><a href="https://github.com/itdxer/neupy/tree/master/examples/boltzmann_machine/rbm_mnist.py">Feature Learning from the MNIST Images</a></li>
                <li><a href="https://github.com/itdxer/neupy/tree/master/examples/boltzmann_machine/rbm_faces_sampling.py">Gibbs sampling using face images</a></li>
              </ul>
            </td>
        </tr>
    </table>

.. raw:: html

    <table border="1" style="background-color: white; margin-top: 50px;">
        <tr>
          <td colspan="2" style="text-align: center;">
              <h3>Competitive networks</h3>
          </td>
        </tr>
        <tr>
            <td width="35%" valign="top" style="text-align: center; padding: 10px;">
                <h3>Growing Neural Gas (GNG)</h3>
                <a href="http://neupy.com/2018/03/26/making_art_with_growing_neural_gas.html#id1">
                <img src="https://raw.githubusercontent.com/itdxer/neupy/master/site/_static/intro/gng-animation-intro.gif" width="100%">
                </a>
            </td>
            <td valign="top" style="padding: 10px;">
                <p>Growing Neural Gas is an algorithm that learns topological structure of the data.</p>

                <ul>
                  <li><a href="https://github.com/itdxer/neupy/blob/master/notebooks/growing-neural-gas/Growing%20Neural%20Gas%20animated.ipynb">Code that makes animation for GNG training</a></li>
                  <li><a href="https://github.com/itdxer/neupy/blob/master/notebooks/growing-neural-gas/Making%20Art%20with%20Growing%20Neural%20Gas.ipynb">Making Art with GNG</a></li>
                </ul>
            </td>
        </tr>
        <tr>
            <td width="35%" valign="top" style="text-align: center; padding: 10px;">
                <h3>Self-Organizing Feature Maps (SOFM or SOM)</h3>
                <img src="https://raw.githubusercontent.com/itdxer/neupy/master/site/_static/intro/sofm-art-intro.png" width="100%">
            </td>
            <td valign="top" style="padding: 10px;">
                <div><b>Notebooks:</b></div>
                <ul>
                  <li><a href="https://github.com/itdxer/neupy/blob/master/notebooks/sofm/Generating%20NeuPy%20logo%20with%20SOFM.ipynb">Creating unique text style</a></li>
                  <li><a href="https://github.com/itdxer/neupy/blob/master/notebooks/sofm/The%20Art%20of%20SOFM.ipynb">Generating Art with SOFM</a></li>
                  <li><a href="https://github.com/itdxer/neupy/blob/master/notebooks/Looking%20inside%20of%20the%20VGG19%20using%20SOFM.ipynb">Visualising VGG19 using SOFM</a></li>
                </ul>
                <div><b>Basics:</b></div>
                <ul>
                  <li><a href="https://github.com/itdxer/neupy/tree/master/examples/competitive/sofm_basic.py">Clustering small dataset</a></li>
                  <li><a href="https://github.com/itdxer/neupy/tree/master/examples/competitive/sofm_iris_clustering.py">Clustering iris dataset</a></li>
                  <li><a href="https://github.com/itdxer/neupy/tree/master/examples/competitive/sofm_moon_topology.py">Learning half-circle topology</a></li>
                  <li><a href="https://github.com/itdxer/neupy/tree/master/examples/competitive/sofm_compare_weight_init.py">Comparison between different weight initialization methods</a></li>
                  <li><a href="https://github.com/itdxer/neupy/tree/master/examples/competitive/sofm_compare_grid_types.py">Comparison between different grid types</a></li>
                </ul>
                <div><b>Advanced:</b></div>
                <ul>
                  <li><a href="https://github.com/itdxer/neupy/tree/master/examples/competitive/sofm_digits.py">Visualize digit images in 2D space with SOFM</a></li>
                  <li><a href="https://github.com/itdxer/neupy/tree/master/examples/competitive/sofm_heatmap_visualization.py">Embedding 30-dimensional dataset into 2D and building heatmap visualization</a></li>
                </ul>
            </td>
        </tr>
        <tr>
            <td width="35%" valign="top" style="text-align: center; padding: 10px;">
                <h3>Linear Vector Quantization (LVQ)</h3>
            </td>
            <td valign="top" style="padding: 10px;">
            <ul>
              <li><a href="https://github.com/itdxer/neupy/tree/master/examples/competitive/reduce_iris_sample_size_lvq.py">Reduce number of training samples in iris dataset with LVQ3</a></li>
            </td>
        </tr>
    </table>

.. raw:: html

    <table border="1" style="background-color: white; margin-top: 50px;">
        <tr>
          <td colspan="2" style="text-align: center;">
              <h3>Associative Memory</h3>
          </td>
        </tr>
        <tr>
            <td width="35%" valign="top" style="text-align: center; padding: 10px;">
                <h3>Discrete Hopfield Neural Network</h3>
                <img src="../_static/docimg/hopfiled-weights.png" width="100%">
            </td>
            <td valign="top" style="padding: 10px;">
                <p>Discrete Hopfield Neural Networks can memorize patterns and reconstruct them from the corrupted samples.</p>

                <div><b>Articles:</b></div>
                <ul>
                  <li><a href="http://neupy.com/2015/09/20/discrete_hopfield_network.html">Exhaustive explanation with example</a></li>
                  <li><a href="http://neupy.com/2015/09/21/password_recovery.html">Password recovery</a></li>
                </ul>

                <div><b>Code:</b></div>
                <ul>
                  <li><a href="https://github.com/itdxer/neupy/tree/master/examples/memory/dhn_energy_func.py">Energy function visualization</a></li>
                  <li><a href="https://github.com/itdxer/neupy/tree/master/examples/memory/password_recovery.py">Password recovery</a></li>
                </ul>
            </td>
        </tr>
        <tr>
            <td width="35%" valign="top" style="text-align: center; padding: 10px;">
                <h3>Cerebellar Model Articulation Controller (CMAC)</h3>
                <img src="../_static/docimg/cmac-sine-func.png" width="100%">
            </td>
            <td valign="top" style="padding: 10px;">
                <p>Cerebellar Model Articulation Controller (CMAC) can quantize continuous space and store it inside of the memory. It's primarily used in the control systems.</p>

                <ul>
                  <li><a href="https://github.com/itdxer/neupy/blob/master/examples/memory/cmac_basic.py">Sine function approximation</a></li>
                </ul>
            </td>
        </tr>
    </table>

.. raw:: html

    <table border="1" style="background-color: white; margin-top: 50px;">
        <tr>
          <td colspan="2" style="text-align: center;">
              <h3>Radial Basis Functions (RBF)</h3>
          </td>
        </tr>
        <tr>
            <td width="35%" valign="top" style="text-align: center; padding: 10px;">
                <h3>Probabilistic Neural Network (PNN)</h3>
            </td>
            <td valign="top" style="padding: 10px;">
                <ul>
                  <li><a href="https://github.com/itdxer/neupy/tree/master/examples/rbfn/music_speech">Music-Speech audio classification</a></li>
                  <li><a href="https://github.com/itdxer/neupy/tree/master/examples/rbfn/pnn_iris.py">Iris dataset classification</a></li>
                </ul>
            </td>
        </tr>
        <tr>
            <td width="35%" valign="top" style="text-align: center; padding: 10px;">
                <h3>Generalized Neural Nerwork (GRNN)</h3>
            </td>
            <td valign="top" style="padding: 10px;">
                <ul>
                  <li><a href="https://github.com/itdxer/neupy/tree/master/examples/rbfn/grnn_params_selection.py">Regression using Diabetes dataset</a></li>
                </ul>
            </td>
        </tr>
    </table>
