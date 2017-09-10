from neupy import layers


__all__ = ('resnet50',)


def ResidualUnit(n_input_filters, n_output_filters, stride,
                 has_branch=False, name=None):

    def bn_name(index):
        return 'bn' + name + '_branch' + index

    def conv_name(index):
        return 'res' + name + '_branch' + index

    main_branch = layers.join(
        # The main purpose of this 1x1 covolutional layer is to
        # reduce number of filters. For instance, for the tensor with
        # 256 filters it can be reduced to 64 reducing amount of
        # computation by factor of 4.
        layers.Convolution(
            size=(n_input_filters, 1, 1),
            stride=stride,
            bias=None,
            name=conv_name('2a'),
        ),
        layers.BatchNorm(name=bn_name('2a')),
        layers.Relu(),

        # This convolutional layer applies 3x3 filter in order to
        # extract features.
        layers.Convolution(
            (n_input_filters, 3, 3),
            padding=1,
            bias=None,
            name=conv_name('2b'),
        ),
        layers.BatchNorm(name=bn_name('2b')),
        layers.Relu(),

        # last layer reverse operations of the first layer. In this
        # case we increase number of filters. For instamce, from previously
        # obtained 64 filters we can increase it back to the 256 filters
        layers.Convolution(
            (n_output_filters, 1, 1),
            bias=None,
            name=conv_name('2c')
        ),
        layers.BatchNorm(name=bn_name('2c')),
    )

    if has_branch:
        residual_branch = layers.join(
            layers.Convolution(
                (n_output_filters, 1, 1),
                stride=stride,
                bias=None,
                name=conv_name('1'),
            ),
            layers.BatchNorm(name=bn_name('1')),
        )
    else:
        # Empty list defines residual connection, meaning that
        # output from this branch would be equal to its input
        residual_branch = []

    return layers.join(
        # For the output from two branches we just combine results
        # with simple elementwise sum operation. The main purpose of
        # the residual connection is to build shortcuts for the
        # gradient during backpropagation.
        [main_branch, residual_branch],
        layers.Elementwise(),
        layers.Relu(),
    )


def resnet50():
    """
    ResNet50 network architecture with random parameters. Parameters
    can be loaded using ``neupy.storage`` module.

    ResNet50 has roughly 25.5 million parameters.

    Notes
    -----
    Because of the global pooling layer, ResNet50 can be applied to
    the images with variable sizes. The only limitation is that image
    size should be bigger than 32x32, otherwise network wont be able
    to apply all transformations to the image.

    Examples
    --------
    >>> from neupy import architectures
    >>> resnet50 = architectures.resnet50()
    >>> resnet50
    (3, 224, 224) -> [... 187 layers ...] -> 1000
    >>>
    >>> from neupy import algorithms
    >>> network = algorithms.Momentum(resnet50)

    See Also
    --------
    :architecture:`vgg16` : VGG16 network
    :architecture:`squeezenet` : SqueezeNet network
    :architecture:`alexnet` : AlexNet network
    :architecture:`squeezenet` : SqueezeNet network

    References
    ----------
    Deep Residual Learning for Image Recognition.
    https://arxiv.org/abs/1512.03385
    """
    return layers.join(
        layers.Input((3, 224, 224)),

        # Convolutional layer reduces image's height and width by a factor
        # of 2 (because of the stride)
        # from (3, 224, 224) to (64, 112, 112)
        layers.Convolution((64, 7, 7), stride=2, padding=3, name='conv1'),
        layers.BatchNorm(name='bn_conv1'),
        layers.Relu(),

        # Stride equal two 2 reduces image size by a factor of two
        # from (64, 112, 112) to (64, 56, 56)
        layers.MaxPooling((3, 3), stride=2, ignore_border=False),

        # The branch option applies extra convolution + batch
        # normalization transforamtions to the residual
        ResidualUnit(64, 256, stride=1, name='2a', has_branch=True),
        ResidualUnit(64, 256, stride=1, name='2b'),
        ResidualUnit(64, 256, stride=1, name='2c'),

        # Another stride=2 reduces width and hight by factor of 2
        ResidualUnit(128, 512, stride=2, name='3a', has_branch=True),
        ResidualUnit(128, 512, stride=1, name='3b'),
        ResidualUnit(128, 512, stride=1, name='3c'),
        ResidualUnit(128, 512, stride=1, name='3d'),

        # Another stride=2 reduces width and hight by factor of 2
        ResidualUnit(256, 1024, stride=2, name='4a', has_branch=True),
        ResidualUnit(256, 1024, stride=1, name='4b'),
        ResidualUnit(256, 1024, stride=1, name='4c'),
        ResidualUnit(256, 1024, stride=1, name='4d'),
        ResidualUnit(256, 1024, stride=1, name='4e'),
        ResidualUnit(256, 1024, stride=1, name='4f'),

        # Another stride=2 reduces width and hight by factor of 2
        ResidualUnit(512, 2048, stride=2, name='5a', has_branch=True),
        ResidualUnit(512, 2048, stride=1, name='5b'),
        ResidualUnit(512, 2048, stride=1, name='5c'),

        # Since the final residual unit has 2048 output filters, global
        # pooling will replace every output image with single average value.
        # Despite input iamge size output from this layer always will be
        # vector with 2048 values
        layers.GlobalPooling(),
        layers.Softmax(1000, name='fc1000'),
    )
