import theano
import theano.tensor as T
from neupy import layers, plots


theano.config.floatX = 'float32'


def Inception(nfilters):
    return layers.join(
        [[
            layers.MaxPooling((3, 3), stride=1, padding=(1, 1)),
            layers.Convolution((nfilters[0], 1, 1)),
            layers.Relu(),
        ], [
            layers.Convolution((nfilters[1], 1, 1)),
            layers.Relu(),
        ], [
            layers.Convolution((nfilters[2], 1, 1)),
            layers.Relu(),
            layers.Convolution((nfilters[3], 3, 3), padding='half'),
            layers.Relu(),
        ], [
            layers.Convolution((nfilters[4], 1, 1)),
            layers.Relu(),
            layers.Convolution((nfilters[5], 5, 5), padding='half'),
            layers.Relu(),
        ]],
        layers.Concatenate(),
    )


googlenet = layers.join(
    layers.Input((3, None, None)),

    layers.Convolution((64, 7, 7), padding='half', stride=2),
    layers.Relu(),
    layers.MaxPooling((3, 3), stride=2),
    layers.LocalResponseNorm(alpha=0.00002, k=1),

    layers.Convolution((64, 1, 1)) > layers.Relu(),
    layers.Convolution((192, 3, 3), padding='half') > layers.Relu(),
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
    layers.GlobalPooling(function=T.mean),

    layers.Softmax(1000),
)
plots.layer_structure(googlenet)
