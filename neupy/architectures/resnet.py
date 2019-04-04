from neupy import layers
from neupy.utils import function_name_scope


__all__ = ('resnet50',)


@function_name_scope
def ResidualUnit(n_input_filters, stride=1, rate=1, has_branch=False,
                 name=None):

    def bn_name(index):
        return 'bn' + name + '_branch' + index

    def conv_name(index):
        return 'res' + name + '_branch' + index

    n_output_filters = 4 * n_input_filters
    main_branch = layers.join(
        # The main purpose of this 1x1 convolution layer is to
        # reduce number of filters. For instance, for the tensor with
        # 256 filters it can be reduced to 64. This trick allows to
        # reduce computation by factor of 4.
        layers.Convolution(
            size=(1, 1, n_input_filters),
            stride=stride,
            bias=None,
            name=conv_name('2a'),
        ),
        layers.BatchNorm(name=bn_name('2a')),
        layers.Relu(),

        # This convolution layer applies 3x3 filter in order to
        # extract features.
        layers.Convolution(
            (3, 3, n_input_filters),
            padding='same',
            dilation=rate,
            bias=None,
            name=conv_name('2b'),
        ),
        layers.BatchNorm(name=bn_name('2b')),
        layers.Relu(),

        # Last layer reverses operations of the first layer. In this
        # case we increase number of filters. For instance, from previously
        # obtained 64 filters we can increase it back to the 256 filters
        layers.Convolution(
            (1, 1, n_output_filters),
            bias=None,
            name=conv_name('2c')
        ),
        layers.BatchNorm(name=bn_name('2c')),
    )

    if has_branch:
        residual_branch = layers.join(
            layers.Convolution(
                (1, 1, n_output_filters),
                stride=stride,
                bias=None,
                name=conv_name('1'),
            ),
            layers.BatchNorm(name=bn_name('1')),
        )
    else:
        # Empty list defines residual connection, meaning that
        # output from this branch would be equal to its input
        residual_branch = layers.Identity('residual-' + name)

    return layers.join(
        # For the output from two branches we just combine results
        # with simple elementwise sum operation. The main purpose of
        # the residual connection is to build shortcuts for the
        # gradient during backpropagation.
        (main_branch | residual_branch),
        layers.Elementwise(),
        layers.Relu(),
    )


def resnet50(input_shape=(224, 224, 3), include_global_pool=True,
             in_out_ratio=32):
    """
    ResNet50 network architecture with random parameters. Parameters
    can be loaded using ``neupy.storage`` module.

    ResNet50 has roughly 25.5 million parameters.

    Parameters
    ----------
    input_shape : tuple
        Network's input shape. Defaults to ``(224, 224, 3)``.

    include_global_pool : bool
        Specifies if returned output should include global pooling
        layer. Defaults to ``True``.

    in_out_ratio : {4, 8, 16, 32}
        Every layer that applies strides reduces height and width per every
        image. There are 5 of these layers in Resnet and at the end each
        dimensions gets reduced by ``32``. For example, 224x224 image
        will be reduced to 7x7 image patches. This parameter specifies
        what level of reduction we want to obtain after we've propagated
        network through all the convolution layers.

    Notes
    -----
    Because of the global pooling layer, ResNet50 can be applied to
    the images with variable sizes. The only limitation is that image
    size should be bigger than 32x32, otherwise network won't be able
    to apply all transformations to the image.

    Examples
    --------
    ResNet-50 for ImageNet classification

    >>> from neupy import architectures, algorithms
    >>>
    >>> resnet = architectures.resnet50()
    >>> resnet
    (?, 224, 224, 3) -> [... 187 layers ...] -> (?, 1000)
    >>>
    >>> optimizer = algorithms.Momentum(resnet50)

    ResNet-50 for custom classification task

    >>> from neupy import architectures
    >>> resnet = architectures.resnet50(include_global_pool=False)
    >>> resnet
    (?, 224, 224, 3) -> [... 185 layers ...] -> (?, 7, 7, 2048)
    >>>
    >>> from neupy.layers import *
    >>> resnet = resnet >> GlobalPooling('avg') >> Softmax(21)
    (?, 224, 224, 3) -> [... 187 layers ...] -> (?, 21)

    ResNet-50 for image segmentation

    >>> from neupy import architectures
    >>> resnet = architectures.resnet50(
    ...     include_global_pool=False,
    ...     in_out_ratio=8,
    ... )
    >>> resnet
    (?, 224, 224, 3) -> [... 185 layers ...] -> (?, 28, 28, 2048)

    See Also
    --------
    :architecture:`vgg16` : VGG16 network
    :architecture:`squeezenet` : SqueezeNet network
    :architecture:`resnet50` : ResNet-50 network

    References
    ----------
    Deep Residual Learning for Image Recognition.
    https://arxiv.org/abs/1512.03385
    """
    in_out_configs = {
        4: {'strides': [1, 1, 1], 'rates': [2, 4, 8]},
        8: {'strides': [2, 1, 1], 'rates': [1, 2, 4]},
        16: {'strides': [2, 2, 1], 'rates': [1, 1, 2]},
        32: {'strides': [2, 2, 2], 'rates': [1, 1, 1]},
    }

    if in_out_ratio not in in_out_configs:
        raise ValueError(
            "Expected one of the folowing in_out_ratio values: {}, got "
            "{} instead.".format(in_out_configs.keys(), in_out_ratio))

    strides = in_out_configs[in_out_ratio]['strides']
    rates = in_out_configs[in_out_ratio]['rates']

    resnet = layers.join(
        layers.Input(input_shape),

        # Convolutional layer reduces image's height and width by a factor
        # of 2 (because of the stride)
        # from (3, 224, 224) to (64, 112, 112)
        layers.Convolution(
            (7, 7, 64), stride=2, bias=None,
            padding='same', name='conv1'
        ),
        layers.BatchNorm(name='bn_conv1'),
        layers.Relu(),

        # Stride equal two 2 reduces image size by a factor of two
        # from (64, 112, 112) to (64, 56, 56)
        layers.MaxPooling((3, 3), stride=2, padding="same"),

        # The branch option applies extra convolution x+ batch
        # normalization transformations to the residual
        ResidualUnit(64, name='2a', has_branch=True),
        ResidualUnit(64, name='2b'),
        ResidualUnit(64, name='2c'),

        # When stride=2 reduces width and hight by factor of 2
        ResidualUnit(128, stride=strides[0], name='3a', has_branch=True),
        ResidualUnit(128, rate=rates[0], name='3b'),
        ResidualUnit(128, rate=rates[0], name='3c'),
        ResidualUnit(128, rate=rates[0], name='3d'),

        # When stride=2 reduces width and hight by factor of 2
        ResidualUnit(256, rate=rates[0], name='4a',
                     stride=strides[1], has_branch=True),
        ResidualUnit(256, rate=rates[1], name='4b'),
        ResidualUnit(256, rate=rates[1], name='4c'),
        ResidualUnit(256, rate=rates[1], name='4d'),
        ResidualUnit(256, rate=rates[1], name='4e'),
        ResidualUnit(256, rate=rates[1], name='4f'),

        # When stride=2 reduces width and hight by factor of 2
        ResidualUnit(512, rate=rates[1], name='5a',
                     stride=strides[2], has_branch=True),
        ResidualUnit(512, rate=rates[2], name='5b'),
        ResidualUnit(512, rate=rates[2], name='5c'),
    )

    if include_global_pool:
        resnet = layers.join(
            resnet,
            # Since the final residual unit has 2048 output filters, global
            # pooling will replace every output image with single average
            # value. Despite input image size, output from this layer always
            # will be a vector with 2048 values.
            layers.GlobalPooling('avg'),
            layers.Softmax(1000, name='fc1000'),
        )

    return resnet
