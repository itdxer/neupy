from neupy import layers


__all__ = ('alexnet',)


class SliceChannels(layers.BaseLayer):
    def __init__(self, from_channel, to_channel):
        self.from_channel = from_channel
        self.to_channel = to_channel
        super(SliceChannels, self).__init__()

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
