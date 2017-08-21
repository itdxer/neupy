from neupy import layers


__all__ = ('alexnet',)


class SliceChannels(layers.BaseLayer):
    """
    Layer expects image as an input with second dimension
    sepcified as a channel. Image will be sliced over the channel.
    The ``from`` and ``to`` indeces can be specified as the parameters.

    Parameters
    ----------
    from_channel : int
        From which channel we will start slicing.

    to_channel : int
        To which channel we will be slicing. This layer won't be
        included in the output.

    {BaseLayer.Parameters}
    """
    def __init__(self, from_channel, to_channel, **kwargs):
        self.from_channel = from_channel
        self.to_channel = to_channel
        super(SliceChannels, self).__init__(**kwargs)

    @property
    def output_shape(self):
        if not self.input_shape:
            return

        _, height, width = self.input_shape
        n_channels = self.to_channel - self.from_channel

        return (n_channels, height, width)

    def output(self, input_value):
        return input_value[:, self.from_channel:self.to_channel, :, :]

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__,
            self.from_channel,
            self.to_channel)


def alexnet():
    """
    AlexNet network architecture with random parameters. Parameters
    can be loaded using ``neupy.storage`` module.

    Originally AlexNet was built in order to solve image classification
    problem. It was used in the ImageNet competition. The goal of the
    competition is to build a model that classifies image into one of
    the 1,000 categories. Categories include animals, objects, transports
    and so on.

    AlexNet has roughly 61 million parameters.

    Examples
    --------
    >>> from neupy import architectures
    >>> alexnet = architectures.alexnet()
    >>> alexnet
    (3, 227, 227) -> [... 37 layers ...] -> 1000
    >>>
    >>> from neupy import algorithms
    >>> network = algorithms.Momentum(alexnet)

    See Also
    --------
    :architecture:`vgg16` : VGG16 network
    :architecture:`vgg19` : VGG19 network
    :architecture:`squeezenet` : SqueezeNet network
    :architecture:`resnet50` : ResNet50 network

    References
    ----------
    ImageNet Classification with Deep Convolutional Neural Networks
    https://goo.gl/479oZZ
    """
    return layers.join(
        layers.Input((3, 227, 227)),

        layers.Convolution((96, 11, 11), stride=(4, 4), name='conv_1'),
        layers.Relu(),

        layers.MaxPooling((3, 3), stride=(2, 2)),
        layers.LocalResponseNorm(),

        [[
            SliceChannels(0, 48),
            layers.Convolution((128, 5, 5), padding=2, name='conv_2_1'),
            layers.Relu(),
        ], [
            SliceChannels(48, 96),
            layers.Convolution((128, 5, 5), padding=2, name='conv_2_2'),
            layers.Relu(),
        ]],
        layers.Concatenate(),

        layers.MaxPooling((3, 3), stride=(2, 2)),
        layers.LocalResponseNorm(),

        layers.Convolution((384, 3, 3), padding=1, name='conv_3'),
        layers.Relu(),

        [[
            SliceChannels(0, 192),
            layers.Convolution((192, 3, 3), padding=1, name='conv_4_1'),
            layers.Relu(),
        ], [
            SliceChannels(192, 384),
            layers.Convolution((192, 3, 3), padding=1, name='conv_4_2'),
            layers.Relu(),
        ]],
        layers.Concatenate(),

        [[
            SliceChannels(0, 192),
            layers.Convolution((128, 3, 3), padding=1, name='conv_5_1'),
            layers.Relu(),
        ], [
            SliceChannels(192, 384),
            layers.Convolution((128, 3, 3), padding=1, name='conv_5_2'),
            layers.Relu(),
        ]],
        layers.Concatenate(),

        layers.MaxPooling((3, 3), stride=(2, 2)),

        layers.Reshape(),
        layers.Relu(4096, name='dense_1') > layers.Dropout(0.5),
        layers.Relu(4096, name='dense_2') > layers.Dropout(0.5),
        layers.Softmax(1000, name='dense_3'),
    )
