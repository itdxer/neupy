import os

import h5py
import theano
import theano.tensor as T
from neupy import layers

from imagenet_tools import (CURRENT_DIR, FILES_DIR, load_image, print_top_n,
                            download_file, extract_params, set_parameters)


theano.config.floatX = 'float32'
VGG16_WEIGHTS_FILE = os.path.join(FILES_DIR, 'vgg16_weights.h5')

vgg16 = layers.join(
    layers.Input((3, 224, 224)),

    layers.Convolution((64, 3, 3), padding=1) > layers.Relu(),
    layers.Convolution((64, 3, 3), padding=1) > layers.Relu(),
    layers.MaxPooling((2, 2)),

    layers.Convolution((128, 3, 3), padding=1) > layers.Relu(),
    layers.Convolution((128, 3, 3), padding=1) > layers.Relu(),
    layers.MaxPooling((2, 2)),

    layers.Convolution((256, 3, 3), padding=1) > layers.Relu(),
    layers.Convolution((256, 3, 3), padding=1) > layers.Relu(),
    layers.Convolution((256, 3, 3), padding=1) > layers.Relu(),
    layers.MaxPooling((2, 2)),

    layers.Convolution((512, 3, 3), padding=1) > layers.Relu(),
    layers.Convolution((512, 3, 3), padding=1) > layers.Relu(),
    layers.Convolution((512, 3, 3), padding=1) > layers.Relu(),
    layers.MaxPooling((2, 2)),

    layers.Convolution((512, 3, 3), padding=1) > layers.Relu(),
    layers.Convolution((512, 3, 3), padding=1) > layers.Relu(),
    layers.Convolution((512, 3, 3), padding=1) > layers.Relu(),
    layers.MaxPooling((2, 2)),

    layers.Reshape(),

    layers.Relu(4096) > layers.Dropout(0.5),
    layers.Relu(4096) > layers.Dropout(0.5),
    layers.Softmax(1000),
)

if not os.path.exists(VGG16_WEIGHTS_FILE):
    download_file(
        url="http://files.heuritech.com/weights/vgg16_weights.h5",
        filepath=VGG16_WEIGHTS_FILE,
        description='Downloading weights'
    )

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

set_parameters(vgg16, parameters)

dog_image = load_image(os.path.join(CURRENT_DIR, 'images', 'dog.jpg'),
                       image_size=(256, 256),
                       crop_size=(224, 224))

# Disables dropout layer
with vgg16.disable_training_state():
    x = T.tensor4()
    predict = theano.function([x], vgg16.output(x))

output = predict(dog_image)
print_top_n(output[0], n=5)
