Train network on GPU
====================

NeuPy is based on the `Theano <http://deeplearning.net/software/theano/>`_ framework and which means that you can easily train neural networks with constructible architectures on GPU.

To learn more about how GPU works on Theano you can check the `this tutorial <http://deeplearning.net/software/theano/tutorial/using_gpu.html>`_.

The simples way to train netwrok on GPU is to run script with the following flag specified at the beggining of the command.

.. code-block:: bash

    $ THEANO_FLAGS="device=cuda0" python train_network.py

Where ``cuda0`` is the name of your device. It can be different value on different machines.
