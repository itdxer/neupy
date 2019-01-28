Transfer Learning
=================

Transfer learning allows us to transfer knowledge learned from one network to different network that aims to solve similar problem. There are two main advantages for this. First, we don't need to start learning from scratch using some random parameters and since we don't start from scratch training should be faster. Second, training on the small datasets might be problematic and prune to overfitting, but with pre-trained layers we can transfer knowledge that has been extracted from large dataset.

In this part of the documentation, we can see how transfer learning can be used in NeuPy. As an example, we can build classifier that expects bird image as an input and it classifies it as one of the 10 bird species. For this task, we can use VGG16 network which was pre-trained using ImageNet data. First, we can use architecture that has been already defined in NeuPy library.

.. code-block:: python

    from neupy import architectures
    from neupy.layers import *
    # At this point vgg16 has only random parameters
    vgg16 = architectures.vgg16()

When we load it by default it has randomly generated parameters. We can load pre-trained parameters using ``neupy.storage`` module. You can download pre-trained models from :ref:`model-zoo`.

.. code-block:: python

    from neupy import storage
    storage.load(vgg16, '/path/to/vgg16.hdf5')

We can check what input and output shapes network expects.

.. code-block:: python

    >>> vgg16
    (?, 224, 224, 3) -> [... 41 layers ...] -> (?, 1000)
    >>> vgg16.layers[-3:]  # check last 3 layers
    [Dropout(proba=0.5, name='dropout-2'),
     Linear(1000, weight=HeNormal(gain=1.0), bias=Constant(0), name='dense_3'),
     Softmax(name='softmax-3')]

Another way to visualize structure of the network is to use ``show`` method.

.. code-block:: python

    >>> vgg16.show()

In both cases, we can see that final layer layer makes prediction for 1,000 classes, but in our classification problem we have only 10 classes. In NeuPy, you can easily slice over the network in order to cut layers that 23 don't need. If you visualized network using the ``plots.network_structure`` function than you should have noticed that it has dropout layer before the final layer. We can use it as a reference point for slicing.

.. code-block:: python

    >>> layer_before_dropout = vgg16.layers[-4]
    >>> vgg16.end(layer_before_dropout)
    (?, 224, 224, 3) -> [... 39 layers ...] -> (?, 4096)
    >>>
    >>> vgg16_modified = vgg16.end(layer_before_dropout) >> Softmax(10)
    >>> vgg16_modified
    (?, 224, 224, 3) -> [... 40 layers ...] -> (?, 10)

The ``vgg16_modified`` network has new architecture that re-uses all layers except the last one from VGG16 architecture and combines it with new ``Softmax(10)`` layer that we added specifically for out bird classifier.

In order to speed up transfer learning, we can exclude transferred layers from the training. They already has been pre-trained for us and in some applications we can just use them without modification. It can give us significant speed up in training time. For this problem we have to explicitly separate our architecture into two different parts. First one should have pre-trained layers and the other one new layers with randomly generated weights.

.. code-block:: python

    >>> pretrained_vgg16_part = vgg16.end(layer_before_dropout)
    >>> pretrained_vgg16_part
    (?, 224, 224, 3) -> [... 38 layers ...] -> (?, 4096)
    >>> new_vgg16_part = Input(4096) >> Softmax(10)

You can notice that for the last layer we create small by network adding ``Input(4096)`` layer. In this way, we're saying that network expects input with 4096 features. It's exactly the same number of feature that we get if we propagate image through pre-trained part of the VGG16. We can transform our input images into vectors with 4096 features after propagating through the pre-trained VGG16 layers. We do it in order to speed up training for the last layer and avoid training for the pre-trained layers. We will use embedded features (4096-dimensional) that we get per each image and our training data for the new layers that we added for our bird classifier.

.. code-block:: python

    >>> from neupy import algorithms
    >>> # Loading 10,000 image that would be pre-processed in the
    >>> # same way as it was done during training on ImageNet data.
    >>> # Labels were encoded with one hot encoder.
    >>> images, targets = load_prepared_image_and_labels()
    >>>
    >>> embedded_images = pretrained_vgg16_part.predict(images, batch_size=128)
    >>> embedded_images.shape
    (10000, 4096)
    >>>
    >>> optimizer = algorithms.Momentum(new_vgg16_part, verbose=True)
    >>> optimizer.train(embedded_images, targets, epochs=1000)

When we finished training, the last layer in the network can be combined with pre-trained VGG16 layers and create full network that we will use to classify birds from images.

.. code-block:: python

    >>> pretrained_vgg16_part >> new_vgg16_part
    (?, 224, 224, 3) -> [... 40 layers ...] -> (?, 10)

Notice, that we still have our ``Input(4096)`` in the ``new_vgg16_part`` network. To make our final architecture cleaner we can simply use only last layer from the ``new_vgg16_part`` network or just use network without first input layer.

If you have enough computational resources and/or you're not satisfied with the accuracy that you get than you can try to remove more layers from the pre-trained network. Also, you can use network that you combined from pre-trained parts and newly trained layer (or multiple layers) and fine-tune layers using the same images, but this time you should use all layers from the network during the training.
