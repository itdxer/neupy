import theano.tensor as T

from neupy import layers


__all__ = ('squeezenet',)


def Fire(s_1x1, e_1x1, e_3x3, name):
    return layers.join(
        layers.Convolution(
            (s_1x1, 1, 1),
            padding='half',
            name=name + '/squeeze1x1'
        ),
        layers.Relu(),
        [[
            layers.Convolution(
                (e_1x1, 1, 1),
                padding='half',
                name=name + '/expand1x1'
            ),
            layers.Relu(),
        ], [
            layers.Convolution(
                (e_3x3, 3, 3),
                padding='half',
                name=name + '/expand3x3'
            ),
            layers.Relu(),
        ]],
        layers.Concatenate(),
    )


def squeezenet():
    """
    SqueezeNet network architecture with random parameters.
    Parameters can be loaded using ``neupy.storage`` module.

    SqueezeNet has roughly 1.2 million parameters. It is almost
    50 times less than in AlexNet. Parameters can be stored as 5Mb
    file.

    Examples
    --------
    >>> from neupy import architectures
    >>> squeezenet = architectures.squeezenet()
    >>> squeezenet
    (3, 227, 227) -> [... 67 layers ...] -> 1000
    >>>
    >>> from neupy import algorithms
    >>> network = algorithms.Momentum(squeezenet)

    See Also
    --------
    :architecture:`vgg16` : VGG16 network
    :architecture:`vgg19` : VGG19 network
    :architecture:`alexnet` : AlexNet network
    :architecture:`resnet50` : ResNet50 network

    References
    ----------
    SqueezeNet: AlexNet-level accuracy with 50x fewer parameters
    and <0.5MB model size
    https://arxiv.org/abs/1602.07360
    """
    return layers.join(
        layers.Input((3, 227, 227)),

        layers.Convolution((96, 3, 3), stride=(2, 2),
                           padding='valid', name='conv1'),
        layers.Relu(),
        layers.MaxPooling((3, 3), stride=(2, 2)),

        Fire(16, 64, 64, name='fire2'),
        Fire(16, 64, 64, name='fire3'),
        Fire(32, 128, 128, name='fire4'),
        layers.MaxPooling((2, 2)),

        Fire(32, 128, 128, name='fire5'),
        Fire(48, 192, 192, name='fire6'),
        Fire(48, 192, 192, name='fire7'),
        Fire(64, 256, 256, name='fire8'),
        layers.MaxPooling((2, 2)),

        Fire(64, 256, 256, name='fire9'),
        layers.Dropout(0.5),

        layers.Convolution((1000, 1, 1), padding='valid', name='conv10'),
        layers.GlobalPooling(function=T.mean),
        layers.Reshape(),
        layers.Softmax(),
    )
