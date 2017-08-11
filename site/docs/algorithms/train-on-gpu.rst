Train network on GPU
====================

NeuPy is based on the `Theano <http://deeplearning.net/software/theano/>`_ framework and it means that you can easily train neural networks with constructible architectures on GPU.

To learn more about how Theano works with GPU works, you can check `this tutorial <http://deeplearning.net/software/theano/tutorial/using_gpu.html>`_.

The simples way to train network on GPU is to run script with the following flag specified at the beginning of the command.

.. code-block:: bash

    $ THEANO_FLAGS="device=cuda0" python train_network.py

Where ``cuda0`` is the name of your GPU device. Value can be different on different machines.
