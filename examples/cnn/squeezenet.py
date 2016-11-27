import theano
import theano.tensor as T
from neupy import layers, plots


theano.config.floatX = 'float32'


def Fire(s_1x1, e_1x1, e_3x3):
    return layers.join(
        layers.Convolution((s_1x1, 1, 1), padding='half'),
        layers.Relu(),
        [[
            layers.Convolution((e_1x1, 1, 1), padding='half'),
            layers.Relu(),
        ], [
            layers.Convolution((e_3x3, 3, 3), padding='half'),
            layers.Relu(),
        ]],
        layers.Concatenate(),
    )


# Networks weight ~4.8 Mb
squeezenet = layers.join(
    layers.Input((3, 224, 224)),

    layers.Convolution((96, 3, 3), stride=(2, 2), padding='valid'),
    layers.Relu(),
    layers.MaxPooling((3, 3), stride=(2, 2)),

    Fire(16, 64, 64),
    Fire(16, 64, 64),
    Fire(32, 128, 128),
    layers.MaxPooling((2, 2)),

    Fire(32, 128, 128),
    Fire(48, 192, 192),
    Fire(48, 192, 192),
    Fire(64, 256, 256),
    layers.MaxPooling((2, 2)),

    Fire(64, 256, 256),
    layers.Dropout(0.5),

    layers.Convolution((1000, 1, 1), padding='valid'),
    layers.GlobalPooling(function=T.mean),
    layers.Reshape(),
    layers.Softmax(),
)
plots.layer_structure(squeezenet)
