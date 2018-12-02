import tensorflow as tf

from neupy import layers, plots


def Inception(nfilters):
    return layers.join(
        [[
            layers.MaxPooling((3, 3), stride=1, padding='SAME'),
            layers.Convolution((1, 1, nfilters[0])),
            layers.Relu(),
        ], [
            layers.Convolution((1, 1, nfilters[1])),
            layers.Relu(),
        ], [
            layers.Convolution((1, 1, nfilters[2])),
            layers.Relu(),
            layers.Convolution((3, 3, nfilters[3]), padding='SAME'),
            layers.Relu(),
        ], [
            layers.Convolution((1, 1, nfilters[4])),
            layers.Relu(),
            layers.Convolution((5, 5, nfilters[5]), padding='SAME'),
            layers.Relu(),
        ]],
        layers.Concatenate(),
    )


UNKNOWN = None
googlenet = layers.join(
    layers.Input((UNKNOWN, UNKNOWN, 3)),

    layers.Convolution((7, 7, 64), padding='SAME', stride=2),
    layers.Relu(),
    layers.MaxPooling((3, 3), stride=2),
    layers.LocalResponseNorm(alpha=0.00002, k=1),

    layers.Convolution((1, 1, 64)) > layers.Relu(),
    layers.Convolution((3, 3, 192), padding='SAME') > layers.Relu(),
    layers.LocalResponseNorm(alpha=0.00002, k=1),
    layers.MaxPooling((3, 3), stride=2),

    Inception((32, 64, 96, 128, 16, 32)),
    Inception((64, 128, 128, 192, 32, 96)),
    layers.MaxPooling((3, 3), stride=2),

    Inception((64, 192, 96, 208, 16, 48)),
    Inception((64, 160, 112, 224, 24, 64)),
    Inception((64, 128, 128, 256, 24, 64)),
    Inception((64, 112, 144, 288, 32, 64)),
    Inception((128, 256, 160, 320, 32, 128)),
    layers.MaxPooling((3, 3), stride=2),

    Inception((128, 256, 160, 320, 32, 128)),
    Inception((128, 384, 192, 384, 48, 128)),
    layers.GlobalPooling(function=tf.reduce_mean),

    layers.Softmax(1000),
)
plots.network_structure(googlenet)
