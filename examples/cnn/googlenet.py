import tensorflow as tf

from neupy.layers import *


def Inception(nfilters):
    return join(
        parallel([
            MaxPooling((3, 3), stride=1, padding='SAME'),
            Convolution((1, 1, nfilters[0])),
            Relu(),
        ], [
            Convolution((1, 1, nfilters[1])),
            Relu(),
        ], [
            Convolution((1, 1, nfilters[2])),
            Relu(),
            Convolution((3, 3, nfilters[3]), padding='SAME'),
            Relu(),
        ], [
            Convolution((1, 1, nfilters[4])),
            Relu(),
            Convolution((5, 5, nfilters[5]), padding='SAME'),
            Relu(),
        ]),
        Concatenate(),
    )


UNKNOWN = None
googlenet = join(
    Input((UNKNOWN, UNKNOWN, 3)),

    Convolution((7, 7, 64), padding='SAME', stride=2),
    Relu(),
    MaxPooling((3, 3), stride=2),
    LocalResponseNorm(alpha=0.00002, k=1),

    Convolution((1, 1, 64)) >> Relu(),
    Convolution((3, 3, 192), padding='SAME') >> Relu(),
    LocalResponseNorm(alpha=0.00002, k=1),
    MaxPooling((3, 3), stride=2),

    Inception((32, 64, 96, 128, 16, 32)),
    Inception((64, 128, 128, 192, 32, 96)),
    MaxPooling((3, 3), stride=2),

    Inception((64, 192, 96, 208, 16, 48)),
    Inception((64, 160, 112, 224, 24, 64)),
    Inception((64, 128, 128, 256, 24, 64)),
    Inception((64, 112, 144, 288, 32, 64)),
    Inception((128, 256, 160, 320, 32, 128)),
    MaxPooling((3, 3), stride=2),

    Inception((128, 256, 160, 320, 32, 128)),
    Inception((128, 384, 192, 384, 48, 128)),
    GlobalPooling('avg'),

    Softmax(1000),
)
googlenet.show()
