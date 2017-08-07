from functools import partial

from neupy import layers


__all__ = ('vgg19',)


def vgg19():
    """
    VGG19 network architecture.

    Examples
    --------
    >>> from neupy import architectures
    >>> vgg19 = architectures.vgg19()
    >>> vgg19
    (3, 224, 224) -> [... 44 layers ...] -> 1000
    >>>
    >>> from neupy import algorithms
    >>> network = algorithms.Momentum(vgg19)

    See Also
    --------
    :architecture:`vgg16` : VGG16 network
    :architecture:`squeezenet` : SqueezeNet network
    :architecture:`alexnet` : AlexNet network
    """
    HalfPadConvolution = partial(layers.Convolution, padding='half')

    return layers.join(
        layers.Input((3, 224, 224)),

        HalfPadConvolution((64, 3, 3), name='conv1_1') > layers.Relu(),
        HalfPadConvolution((64, 3, 3), name='conv1_2') > layers.Relu(),
        layers.MaxPooling((2, 2)),

        HalfPadConvolution((128, 3, 3), name='conv2_1') > layers.Relu(),
        HalfPadConvolution((128, 3, 3), name='conv2_2') > layers.Relu(),
        layers.MaxPooling((2, 2)),

        HalfPadConvolution((256, 3, 3), name='conv3_1') > layers.Relu(),
        HalfPadConvolution((256, 3, 3), name='conv3_2') > layers.Relu(),
        HalfPadConvolution((256, 3, 3), name='conv3_3') > layers.Relu(),
        HalfPadConvolution((256, 3, 3), name='conv3_4') > layers.Relu(),
        layers.MaxPooling((2, 2)),

        HalfPadConvolution((512, 3, 3), name='conv4_1') > layers.Relu(),
        HalfPadConvolution((512, 3, 3), name='conv4_2') > layers.Relu(),
        HalfPadConvolution((512, 3, 3), name='conv4_3') > layers.Relu(),
        HalfPadConvolution((512, 3, 3), name='conv4_4') > layers.Relu(),
        layers.MaxPooling((2, 2)),

        HalfPadConvolution((512, 3, 3), name='conv5_1') > layers.Relu(),
        HalfPadConvolution((512, 3, 3), name='conv5_2') > layers.Relu(),
        HalfPadConvolution((512, 3, 3), name='conv5_3') > layers.Relu(),
        HalfPadConvolution((512, 3, 3), name='conv5_4') > layers.Relu(),
        layers.MaxPooling((2, 2)),

        layers.Reshape(),

        layers.Relu(4096, name='dense_1') > layers.Dropout(0.5),
        layers.Relu(4096, name='dense_2') > layers.Dropout(0.5),
        layers.Softmax(1000, name='dense_3'),
    )
