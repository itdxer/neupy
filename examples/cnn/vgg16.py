import os

import theano
from neupy import layers, storage

from imagenet_tools import (CURRENT_DIR, FILES_DIR, load_image,
                            print_top_n, download_file)


theano.config.floatX = 'float32'
VGG16_WEIGHTS_FILE = os.path.join(FILES_DIR, 'vgg16.pickle')

vgg16 = layers.join(
    layers.Input((3, 224, 224)),

    layers.Convolution((64, 3, 3), padding=1, name='conv1_1') > layers.Relu(),
    layers.Convolution((64, 3, 3), padding=1, name='conv1_2') > layers.Relu(),
    layers.MaxPooling((2, 2)),

    layers.Convolution((128, 3, 3), padding=1, name='conv2_1') > layers.Relu(),
    layers.Convolution((128, 3, 3), padding=1, name='conv2_2') > layers.Relu(),
    layers.MaxPooling((2, 2)),

    layers.Convolution((256, 3, 3), padding=1, name='conv3_1') > layers.Relu(),
    layers.Convolution((256, 3, 3), padding=1, name='conv3_2') > layers.Relu(),
    layers.Convolution((256, 3, 3), padding=1, name='conv3_3') > layers.Relu(),
    layers.MaxPooling((2, 2)),

    layers.Convolution((512, 3, 3), padding=1, name='conv4_1') > layers.Relu(),
    layers.Convolution((512, 3, 3), padding=1, name='conv4_2') > layers.Relu(),
    layers.Convolution((512, 3, 3), padding=1, name='conv4_3') > layers.Relu(),
    layers.MaxPooling((2, 2)),

    layers.Convolution((512, 3, 3), padding=1, name='conv5_1') > layers.Relu(),
    layers.Convolution((512, 3, 3), padding=1, name='conv5_2') > layers.Relu(),
    layers.Convolution((512, 3, 3), padding=1, name='conv5_3') > layers.Relu(),
    layers.MaxPooling((2, 2)),

    layers.Reshape(),

    layers.Relu(4096, name='dense_1') > layers.Dropout(0.5),
    layers.Relu(4096, name='dense_2') > layers.Dropout(0.5),
    layers.Softmax(1000, name='dense_3'),
)

if not os.path.exists(VGG16_WEIGHTS_FILE):
    download_file(
        url=(
            "http://srv70.putdrive.com/putstorage/DownloadFileHash/"
            "5B7DCBF43A5A4A5QQWE2301430EWQS/vgg16.pickle"
        ),
        filepath=VGG16_WEIGHTS_FILE,
        description='Downloading weights'
    )

storage.load(vgg16, VGG16_WEIGHTS_FILE)

dog_image = load_image(
    os.path.join(CURRENT_DIR, 'images', 'dog.jpg'),
    image_size=(256, 256),
    crop_size=(224, 224))

predict = vgg16.compile()
output = predict(dog_image)

print_top_n(output[0], n=5)
