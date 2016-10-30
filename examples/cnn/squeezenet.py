import theano.tensor as T
from neupy import layers


def Fire(s_1x1, e_1x1, e_3x3):
    return layers.join(
        layers.Convolution((s_1x1, 1, 1), border_mode='half'),
        layers.Relu(),

        layers.Parallel(
            [[
                layers.Convolution((e_1x1, 1, 1), border_mode='half'),
                layers.Relu(),
            ], [
                layers.Convolution((e_3x3, 3, 3), border_mode='half'),
                layers.Relu(),
            ]],
            layers.Concatenate(),
        )
    )


squeezenet = layers.join(
    layers.Input((3, 224, 224)),

    layers.Convolution((96, 3, 3), stride_size=(2, 2), border_mode='valid'),
    layers.Relu(),
    layers.MaxPooling((3, 3), stride_size=(2, 2)),

    Fire(s_1x1=16, e_1x1=64, e_3x3=64),
    Fire(s_1x1=16, e_1x1=64, e_3x3=64),
    Fire(s_1x1=32, e_1x1=128, e_3x3=128),
    layers.MaxPooling((2, 2)),

    Fire(s_1x1=32, e_1x1=128, e_3x3=128),
    Fire(s_1x1=48, e_1x1=192, e_3x3=192),
    Fire(s_1x1=48, e_1x1=192, e_3x3=192),
    Fire(s_1x1=64, e_1x1=256, e_3x3=256),
    layers.MaxPooling((2, 2)),

    Fire(64, 256, 256),
    layers.Dropout(0.5),

    layers.Convolution((1000, 1, 1), border_mode='valid'),
    layers.GlobalPooling(function=T.mean),
    layers.Reshape(),
    layers.Softmax(),
)
