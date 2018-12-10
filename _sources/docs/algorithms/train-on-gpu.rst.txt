Train network on GPU
====================

NeuPy is based on the `Tensorfow <https://tensorflow.org/>`_ framework and it means that you can easily train neural networks with constructible architectures on GPU.

Training on GPU doesn't require any modifications in your code. If you want to use specific GPU among available options, you just need to use the ``tf.device`` context manager. See official documentation from tensorflow `here <https://www.tensorflow.org/api_docs/python/tf/device>`_
