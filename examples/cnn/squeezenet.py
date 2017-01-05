import os

import theano
import theano.tensor as T
from neupy import layers, storage

from imagenet_tools import (CURRENT_DIR, FILES_DIR, load_image,
                            print_top_n, download_file)


theano.config.floatX = 'float32'
SQUEEZENET_WEIGHTS_FILE = os.path.join(FILES_DIR, 'squeezenet.pickle')


def Fire(s_1x1, e_1x1, e_3x3, name):
    return layers.join(
        layers.Convolution((s_1x1, 1, 1), padding='half',
                           name=name + '/squeeze1x1'),
        layers.Relu(),
        [[
            layers.Convolution((e_1x1, 1, 1), padding='half',
                               name=name + '/expand1x1'),
            layers.Relu(),
        ], [
            layers.Convolution((e_3x3, 3, 3), padding='half',
                               name=name + '/expand3x3'),
            layers.Relu(),
        ]],
        layers.Concatenate(),
    )


# Networks weight ~4.8 Mb
squeezenet = layers.join(
    layers.Input((3, 224, 224)),

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

if not os.path.exists(SQUEEZENET_WEIGHTS_FILE):
    download_file(
        url=(
            "http://srv70.putdrive.com/putstorage/DownloadFileHash/"
            "6B0A15B43A5A4A5QQWE2304100EWQS/squeezenet.pickle"
        ),
        filepath=SQUEEZENET_WEIGHTS_FILE,
        description='Downloading weights'
    )

storage.load(squeezenet, SQUEEZENET_WEIGHTS_FILE)

monkey_image = load_image(
    os.path.join(CURRENT_DIR, 'images', 'titi-monkey.jpg'),
    image_size=(224, 224))

predict = squeezenet.compile()
output = predict(monkey_image)

print_top_n(output[0], n=5)
