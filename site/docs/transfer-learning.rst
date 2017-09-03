Transfer Learning
=================

The main goal of the transfer learning is to re-use parameters of the network that was trained on one task in order to solve similar problem. For instance, using classifier build on the 1,000 categories from ImageNet we can build similar network that predicts 10 bird species even if they weren't introduced in the ImageNet dataset. It works because network learn low level features that can be re-used for other problems. Transferring parameters can significantly simplify training procedure of the large networks, since most of the parameters have been already pre-trained.

Transfer learning can be easily implemented in NeuPy. For example, we can build classifier that using images predicts 10 different bird species. For this task, we can re-use VGG16 network.

.. code-block:: python

    from neupy import architectures, layers
    # At this point vgg16 has only random parameters
    vgg16 = architectures.vgg16()

At this point VGG16 network has randomly generated parameter that will make uniform predictions for all classes. We can load pre-trained parameters using ``neupy.storage`` module.

.. code-block:: python

    from neupy import storage
    storage.load_pickle(vgg16, '/path/to/vgg16.pickle')

We can check some basic information about network architecture before modifying it.

.. code-block:: python

    >>> vgg16.input_shape
    (3, 224, 224)
    >>> vgg16.output_shape
    (1000,)
    >>> vgg16.layers[-3:]  # check last 3 layers
    [Relu(4096), Dropout(proba=0.5), Softmax(1000)]

As you can see the last layer makes prediction for 1,000 classes and for our problem we need only 10. For this reason, we can replace last layer with layer that specific for our classification problem.

We can use dropout layer as the last layer that we will re-use from VGG16 network.

.. code-block:: python

    >>> dropout = vgg16.layers[-2]
    >>> vgg16.end(dropout)
    (3, 224, 224) -> [... 37 layers ...] -> 4096
    >>>
    >>> vgg16_modified = vgg16.end(dropout) > layers.Softmax(10)
    >>> vgg16_modified
    (3, 224, 224) -> [... 38 layers ...] -> 10

Now network in the ``vgg16_modified`` variables has replaced last layer with new layer.

In case of transfer learning, often, we need to train last layer first before training whole network. In order to do it, we need to separate network into two parts. First part will have only pre-trained parameters from VGG16 and another one will have only parameters that we want to train specifically for our problem.

.. code-block:: python

    >>> pretrained_vgg16_part = vgg16.end(dropout)
    >>> pretrained_vgg16_part
    (3, 224, 224) -> [... 37 layers ...] -> 4096
    >>> pretrained_vgg16_part.output_shape
    (4096,)
    >>>
    >>> new_vgg16_part = layers.Input(4096) > layers.Softmax(10)
    Input(4096) > Softmax(10)

Using pre-trained part of the network we can prepare input image for the training of the final layer

.. code-block:: python

    >>> from neupy import algorithms
    >>> # Loading 10,000 image that would be pre-processed in the
    >>> # same way as it was done during training on ImageNet data.
    >>> # Labels also was encoded with one hot encoder.
    >>> images, targets = load_prepared_image_and_labels()
    >>>
    >>> # Initialize it using training algorithm in order to
    >>> # get some basic information and a nice progressbar
    >>> # during prediction
    >>> gdnet = algorithms.MinibatchGradientDescent(
    ...     pretrained_vgg16_part,
    ...     batch_size=32,
    ...     verbose=True)
    ...
    >>> embedded_images = gdnet.predict(images)
    >>> embedded_images.shape
    (10000, 4096)
    >>>
    >>> momentum = algorithms.Momentum(new_vgg16_part, verbose=True)
    >>> momentum.train(embedded_images, targets, epochs=1000)

After pre-training of the final layer, we can combine two networks into the new one that we can use as a modified version of VGG16 network that would be used for new application.

.. code-block:: python

    >>> pretrained_vgg16_part > new_vgg16_part
    (3, 224, 224) -> [... 39 layers ...] -> 10
