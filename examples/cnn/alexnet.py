import os

import h5py
import theano
import theano.tensor as T
from neupy import layers

from imagenet_tools import (CURRENT_DIR, FILES_DIR, load_image, print_top_n,
                            download_file, extract_params)


theano.config.floatX = 'float32'
ALEXNET_WEIGHTS_FILE = os.path.join(FILES_DIR, 'alexnet_weights.h5')


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
        return "{}({}, {})".format(self.__class__.__name__,
                                   self.from_channel, self.to_channel)


if not os.path.exists(ALEXNET_WEIGHTS_FILE):
    download_file(
        url="http://files.heuritech.com/weights/alexnet_weights.h5",
        filepath=ALEXNET_WEIGHTS_FILE,
        description='Downloading weights'
    )

all_params = h5py.File(ALEXNET_WEIGHTS_FILE, 'r')

conv_1 = extract_params(all_params, 'conv_1')

conv_2_1 = extract_params(all_params, 'conv_2_1')
conv_2_2 = extract_params(all_params, 'conv_2_2')

conv_3 = extract_params(all_params, 'conv_3')

conv_4_1 = extract_params(all_params, 'conv_4_1')
conv_4_2 = extract_params(all_params, 'conv_4_2')

conv_5_1 = extract_params(all_params, 'conv_5_1')
conv_5_2 = extract_params(all_params, 'conv_5_2')

dense_1 = extract_params(all_params, 'dense_1')
dense_2 = extract_params(all_params, 'dense_2')
dense_3 = extract_params(all_params, 'dense_3')

alexnet = layers.join(
    layers.Input((3, 227, 227)),

    layers.Convolution((96, 11, 11), stride_size=(4, 4), **conv_1),
    layers.Relu(),

    layers.MaxPooling((3, 3), stride_size=(2, 2)),
    layers.LocalResponseNorm(),

    layers.Parallel(
        [[
            SliceChannels(0, 48),
            layers.Convolution((128, 5, 5), border_mode=2, **conv_2_1),
            layers.Relu(),
        ], [
            SliceChannels(48, 96),
            layers.Convolution((128, 5, 5), border_mode=2, **conv_2_2),
            layers.Relu(),
        ]],
        layers.Concatenate(),
    ),
    layers.MaxPooling((3, 3), stride_size=(2, 2)),
    layers.LocalResponseNorm(),

    layers.Convolution((384, 3, 3), border_mode=1, **conv_3) > layers.Relu(),

    layers.Parallel(
        [[
            SliceChannels(0, 192),
            layers.Convolution((192, 3, 3), border_mode=1, **conv_4_1),
            layers.Relu(),
        ], [
            SliceChannels(192, 384),
            layers.Convolution((192, 3, 3), border_mode=1, **conv_4_2),
            layers.Relu(),
        ]],
        layers.Concatenate(),
    ),
    layers.Parallel(
        [[
            SliceChannels(0, 192),
            layers.Convolution((128, 3, 3), border_mode=1, **conv_5_1),
            layers.Relu(),
        ], [
            SliceChannels(192, 384),
            layers.Convolution((128, 3, 3), border_mode=1, **conv_5_2),
            layers.Relu(),
        ]],
        layers.Concatenate(),
    ),
    layers.MaxPooling((3, 3), stride_size=(2, 2)),

    layers.Reshape(),
    layers.Relu(4096, **dense_1) > layers.Dropout(0.5),
    layers.Relu(4096, **dense_2) > layers.Dropout(0.5),
    layers.Softmax(1000, **dense_3),
)

dog_image = load_image(os.path.join(CURRENT_DIR, 'images', 'dog.jpg'),
                       image_size=(256, 256),
                       crop_size=(227, 227),
                       use_bgr=False)

# Disables dropout layer
with alexnet.disable_training_state():
    x = T.tensor4()
    predict = theano.function([x], alexnet.output(x))

output = predict(dog_image)
print_top_n(output[0], n=5)
