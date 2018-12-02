from neupy import layers, plots


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

        height, width, _ = self.input_shape
        n_channels = self.to_channel - self.from_channel

        return (height, width, n_channels)

    def output(self, input_value):
        return input_value[:, :, :, self.from_channel:self.to_channel]

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__,
            self.from_channel,
            self.to_channel)


alexnet = layers.join(
    layers.Input((227, 227, 3)),

    layers.Convolution((11, 11, 96), stride=(4, 4), name='conv_1'),
    layers.Relu(),

    layers.MaxPooling((3, 3), stride=(2, 2)),
    layers.LocalResponseNorm(),

    [[
        SliceChannels(0, 48),
        layers.Convolution((5, 5, 128), padding='SAME', name='conv_2_1'),
        layers.Relu(),
    ], [
        SliceChannels(48, 96),
        layers.Convolution((5, 5, 128), padding='SAME', name='conv_2_2'),
        layers.Relu(),
    ]],
    layers.Concatenate(),

    layers.MaxPooling((3, 3), stride=(2, 2)),
    layers.LocalResponseNorm(),

    layers.Convolution((3, 3, 384), padding='SAME', name='conv_3'),
    layers.Relu(),

    [[
        SliceChannels(0, 192),
        layers.Convolution((3, 3, 192), padding='SAME', name='conv_4_1'),
        layers.Relu(),
    ], [
        SliceChannels(192, 384),
        layers.Convolution((3, 3, 192), padding='SAME', name='conv_4_2'),
        layers.Relu(),
    ]],
    layers.Concatenate(),

    [[
        SliceChannels(0, 192),
        layers.Convolution((3, 3, 128), padding='SAME', name='conv_5_1'),
        layers.Relu(),
    ], [
        SliceChannels(192, 384),
        layers.Convolution((3, 3, 128), padding='SAME', name='conv_5_2'),
        layers.Relu(),
    ]],
    layers.Concatenate(),
    layers.MaxPooling((3, 3), stride=(2, 2)),

    layers.Reshape(),
    layers.Relu(4096, name='dense_1') > layers.Dropout(0.5),
    layers.Relu(4096, name='dense_2') > layers.Dropout(0.5),
    layers.Softmax(1000, name='dense_3'),
)
plots.network_structure(alexnet)
