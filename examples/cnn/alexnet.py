import tensorflow as tf

from neupy.layers import *


class SliceChannels(BaseLayer):
    """
    Layer expects image as an input with second dimension
    sepcified as a channel. Image will be sliced over the channel.
    The ``from`` and ``to`` indices can be specified as the parameters.

    Parameters
    ----------
    from_channel : int
        From which channel we will start slicing.

    to_channel : int
        To which channel we will be slicing. This layer won't be
        included in the output.

    {BaseLayer.name}
    """
    def __init__(self, from_channel, to_channel, name=None):
        self.from_channel = from_channel
        self.to_channel = to_channel
        super(SliceChannels, self).__init__(name=name)

    def get_output_shape(self, input_shape):
        n_channels = self.to_channel - self.from_channel

        if input_shape.ndims is None:
            return tf.TensorShape((None, None, None, n_channels))

        if input_shape.ndims != 4:
            raise ValueError(
                "Layer {} expects 4 dimensional inputs, got {} instead."
                "".format(self.name, input_shape.ndims))

        n_samples, height, width, _ = input_shape
        return tf.TensorShape((n_samples, height, width, n_channels))

    def output(self, input_value):
        return input_value[:, :, :, self.from_channel:self.to_channel]

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__,
            self.from_channel,
            self.to_channel)


alexnet = join(
    Input((227, 227, 3)),

    Convolution((11, 11, 96), stride=(4, 4), name='conv_1') >> Relu(),
    MaxPooling((3, 3), stride=(2, 2)),
    LocalResponseNorm(),

    parallel([
        SliceChannels(0, 48),
        Convolution((5, 5, 128), padding='SAME', name='conv_2_1') >> Relu(),
    ], [
        SliceChannels(48, 96),
        Convolution((5, 5, 128), padding='SAME', name='conv_2_2') >> Relu(),
    ]),
    Concatenate(),

    MaxPooling((3, 3), stride=(2, 2)),
    LocalResponseNorm(),
    Convolution((3, 3, 384), padding='SAME', name='conv_3') >> Relu(),

    parallel([
        SliceChannels(0, 192),
        Convolution((3, 3, 192), padding='SAME', name='conv_4_1') >> Relu(),
    ], [
        SliceChannels(192, 384),
        Convolution((3, 3, 192), padding='SAME', name='conv_4_2') >> Relu(),
    ]),
    Concatenate(),

    parallel([
        SliceChannels(0, 192),
        Convolution((3, 3, 128), padding='SAME', name='conv_5_1') >> Relu(),
    ], [
        SliceChannels(192, 384),
        Convolution((3, 3, 128), padding='SAME', name='conv_5_2') >> Relu(),
    ]),
    Concatenate(),
    MaxPooling((3, 3), stride=(2, 2)),

    Reshape(),
    Relu(4096, name='dense_1') >> Dropout(0.5),
    Relu(4096, name='dense_2') >> Dropout(0.5),
    Softmax(1000, name='dense_3'),
)
alexnet.show()
