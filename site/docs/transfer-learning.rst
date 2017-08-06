Transfer Learning
=================

Transfer learning can be easely implemented in NeuPy. As an example, we can replace final layer in VGG16 network with new one that predicts 10 classes instead of 1000.

.. code-block:: python

    from neupy import architectures, layers
    # At this point vgg16 has only random parameters
    vgg16 = architectures.vgg16()

VGG16 network has random parameters specified by default, for this reason we need to load pre-trained parameters first.

    .. code-block:: python

    >>> from neupy import storage
    >>> storage.load_pickle(vgg16, '/path/to/vgg16.pickle')

Next we need to remove only last layer and there are multiple ways to do it. For instance, we can find last layer that we want to keep in our new network.

.. code-block:: python

    >>> vgg16.input_shape
    (3, 224, 224)
    >>> vgg16.output_shape
    (1000,)
    >>> vgg16.layers[-3:]  # check last 3 layers
    [Relu(4096), Dropout(proba=0.5), Softmax(1000)]

We can use dropout layer as the last layer that we will use from VGG16 network.

.. code-block:: python

    >>> dropout = vgg16.layers[-2]
    >>> vgg16.end(dropout)
    (3, 224, 224) -> [... 37 layers ...] -> 4096
    >>>
    >>> vgg16_modified = vgg16.end(dropout) > layers.Softmax(10)
    >>> vgg16_modified
    (3, 224, 224) -> [... 38 layers ...] -> 10

Now network in the ``vgg16_modified`` variables has replaced last layer with new layer.

In case of transfer learning often we need to train last layer first before training whole network. In order to do it we need to separater network into two parts. First parts will have only pretrained parameters from VGG16 and another one will have only parameters that we want to train specificaly for our problem.

.. code-block:: python

    >>> pretrained_vgg16_part = vgg16.end(dropout)
    >>> pretrained_vgg16_part
    (3, 224, 224) -> [... 37 layers ...] -> 4096
    >>> pretrained_vgg16_part.output_shape
    (4096,)
    >>>
    >>> new_vgg16_part = layers.Input(4096) > layers.Softmax(10)
    Input(4096) > Softmax(10)

Usign pretrained part of the network we can prepare input image for the training of the final layer

.. code-block:: python

    >>> from neupy import algorithms
    >>> # Loading 10,000 image that would be preprocesed in the
    >>> # same way as during imagenet training
    >>> # Labels also was encoded with one hot encoder.
    >>> images, targets = load_prepared_image_and_labels()
    >>>
    >>> # Initialize it using training algorithm in order to
    >>> # have a nice progressbar during prediction
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

After pre-training of the final layer we can combine two networks into the new one that we can use as a modified version of VGG16 network that would be used for new application.

.. code-block:: python

    >>> pretrained_vgg16_part > new_vgg16_part
    (3, 224, 224) -> [... 39 layers ...] -> 10
