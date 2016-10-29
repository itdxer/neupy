import os

import h5py
import theano
import theano.tensor as T
from neupy import layers

from imagenet_tools import (CURRENT_DIR, FILES_DIR, load_image, print_top_n,
                            download_file)


theano.config.floatX = 'float32'
VGG16_WEIGHTS_FILE = os.path.join(FILES_DIR, 'vgg16_weights.h5')

vgg16 = layers.join(
    layers.Input((3, 224, 224)),

    layers.Convolution((64, 3, 3), border_mode=1) > layers.Relu(),
    layers.Convolution((64, 3, 3), border_mode=1) > layers.Relu(),
    layers.MaxPooling((2, 2)),

    layers.Convolution((128, 3, 3), border_mode=1) > layers.Relu(),
    layers.Convolution((128, 3, 3), border_mode=1) > layers.Relu(),
    layers.MaxPooling((2, 2)),

    layers.Convolution((256, 3, 3), border_mode=1) > layers.Relu(),
    layers.Convolution((256, 3, 3), border_mode=1) > layers.Relu(),
    layers.Convolution((256, 3, 3), border_mode=1) > layers.Relu(),
    layers.MaxPooling((2, 2)),

    layers.Convolution((512, 3, 3), border_mode=1) > layers.Relu(),
    layers.Convolution((512, 3, 3), border_mode=1) > layers.Relu(),
    layers.Convolution((512, 3, 3), border_mode=1) > layers.Relu(),
    layers.MaxPooling((2, 2)),

    layers.Convolution((512, 3, 3), border_mode=1) > layers.Relu(),
    layers.Convolution((512, 3, 3), border_mode=1) > layers.Relu(),
    layers.Convolution((512, 3, 3), border_mode=1) > layers.Relu(),
    layers.MaxPooling((2, 2)),

    layers.Reshape(),

    layers.Relu(4096) > layers.Dropout(0.5),
    layers.Relu(4096) > layers.Dropout(0.5),
    layers.Softmax(1000),
)


def extract_params(all_params, name):
    params = all_params[name]
    return {
        'weight': params['{}_W'.format(name)].value,
        'bias': params['{}_b'.format(name)].value,
    }


if not os.path.exists(VGG16_WEIGHTS_FILE):
    print('Downloading weights')
    download_file(
        url="http://files.heuritech.com/weights/vgg16_weights.h5",
        filepath=VGG16_WEIGHTS_FILE
    )
    print('Downloaded sucessfully')


all_params = h5py.File(VGG16_WEIGHTS_FILE, 'r')
parameters = [
    extract_params(all_params, 'conv1_1'),
    extract_params(all_params, 'conv1_2'),

    extract_params(all_params, 'conv2_1'),
    extract_params(all_params, 'conv2_2'),

    extract_params(all_params, 'conv3_1'),
    extract_params(all_params, 'conv3_2'),
    extract_params(all_params, 'conv3_3'),

    extract_params(all_params, 'conv4_1'),
    extract_params(all_params, 'conv4_2'),
    extract_params(all_params, 'conv4_3'),

    extract_params(all_params, 'conv5_1'),
    extract_params(all_params, 'conv5_2'),
    extract_params(all_params, 'conv5_3'),

    extract_params(all_params, 'dense_1'),
    extract_params(all_params, 'dense_2'),
    extract_params(all_params, 'dense_3'),
]

for layer in vgg16:
    if layer.parameters:
        new_parameters = parameters.pop(0)
        for param_name, param_value in new_parameters.items():
            layer_param = getattr(layer, param_name)
            layer_param.set_value(param_value)

dog_image = load_image(os.path.join(CURRENT_DIR, 'images', 'dog.jpg'),
                       image_size=(256, 256),
                       crop_size=(224, 224))

# Disables dropout layer
with vgg16.disable_training_state():
    x = T.tensor4()
    predict = theano.function([x], vgg16.output(x))

output = predict(dog_image)
print_top_n(output[0], n=5)
