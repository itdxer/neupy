from functools import partial

from neupy import layers, plots


def ConvReluBN(*conv_args, **conv_kwargs):
    return layers.join(
        layers.Convolution(*conv_args, **conv_kwargs),
        layers.Relu(),
        layers.BatchNorm(epsilon=0.001),
    )


def Inception_1(conv_filters):
    return layers.join(
        [[
            ConvReluBN((conv_filters[0][0], 1, 1)),
        ], [
            ConvReluBN((conv_filters[1][0], 1, 1)),
            ConvReluBN((conv_filters[1][1], 5, 5), padding=2),
        ], [
            ConvReluBN((conv_filters[2][0], 1, 1)),
            ConvReluBN((conv_filters[2][1], 3, 3), padding=1),
            ConvReluBN((conv_filters[2][2], 3, 3), padding=1),
        ], [
            layers.AveragePooling((3, 3), stride=(1, 1), padding=1,
                                  mode='exclude_padding'),
            ConvReluBN((conv_filters[3][0], 1, 1)),
        ]],
        layers.Concatenate(),
    )


def Inception_2(conv_filters):
    return layers.join(
        [[
            ConvReluBN((conv_filters[0][0], 1, 1)),
        ], [
            ConvReluBN((conv_filters[1][0], 1, 1)),
            ConvReluBN((conv_filters[1][1], 1, 7), padding=(0, 3)),
            ConvReluBN((conv_filters[1][2], 7, 1), padding=(3, 0)),
        ], [
            ConvReluBN((conv_filters[2][0], 1, 1)),
            ConvReluBN((conv_filters[2][1], 7, 1), padding=(3, 0)),
            ConvReluBN((conv_filters[2][2], 1, 7), padding=(0, 3)),
            ConvReluBN((conv_filters[2][3], 7, 1), padding=(3, 0)),
            ConvReluBN((conv_filters[2][4], 1, 7), padding=(0, 3)),
        ], [
            layers.AveragePooling((3, 3), stride=(1, 1), padding=1,
                                  mode='exclude_padding'),
            ConvReluBN((conv_filters[3][0], 1, 1)),
        ]],
        layers.Concatenate(),
    )


def Inception_3(pooling):
    if pooling not in ('max', 'average'):
        raise ValueError("Invalid pooling option: {}".format(pooling))

    if pooling == 'max':
        Pooling = layers.MaxPooling

    elif pooling == 'average':
        Pooling = partial(layers.AveragePooling, mode='exclude_padding')

    return layers.join(
        [[
            ConvReluBN((320, 1, 1)),
        ], [
            ConvReluBN((384, 1, 1)),
            [[
                ConvReluBN((384, 1, 3), padding=(0, 1)),
            ], [
                ConvReluBN((384, 3, 1), padding=(1, 0)),
            ]],
        ], [
            ConvReluBN((448, 1, 1)),
            ConvReluBN((384, 3, 3), padding=1),
            [[
                ConvReluBN((384, 1, 3), padding=(0, 1)),
            ], [
                ConvReluBN((384, 3, 1), padding=(1, 0)),
            ]],
        ], [
            Pooling((3, 3), stride=(1, 1), padding=1),
            ConvReluBN((192, 1, 1)),
        ]],
        layers.Concatenate(),
    )


inception_v3 = layers.join(
    layers.Input((3, 299, 299)),

    ConvReluBN((32, 3, 3), stride=2),
    ConvReluBN((32, 3, 3)),
    ConvReluBN((64, 3, 3), padding=1),
    layers.MaxPooling((3, 3), stride=(2, 2)),

    ConvReluBN((80, 1, 1)),
    ConvReluBN((192, 3, 3)),
    layers.MaxPooling((3, 3), stride=(2, 2)),

    Inception_1([[64], [48, 64], [64, 96, 96], [32]]),
    Inception_1([[64], [48, 64], [64, 96, 96], [64]]),
    Inception_1([[64], [48, 64], [64, 96, 96], [64]]),

    [[
        ConvReluBN((384, 3, 3), stride=2),
    ], [
        ConvReluBN((64, 1, 1)),
        ConvReluBN((96, 3, 3), padding=1),
        ConvReluBN((96, 3, 3), stride=2),
    ], [
        layers.MaxPooling((3, 3), stride=(2, 2))
    ]],
    layers.Concatenate(),

    Inception_2([[192], [128, 128, 192], [128, 128, 128, 128, 192], [192]]),
    Inception_2([[192], [160, 160, 192], [160, 160, 160, 160, 192], [192]]),
    Inception_2([[192], [160, 160, 192], [160, 160, 160, 160, 192], [192]]),
    Inception_2([[192], [192, 192, 192], [192, 192, 192, 192, 192], [192]]),

    [[
        ConvReluBN((192, 1, 1)),
        ConvReluBN((320, 3, 3), stride=2),
    ], [
        ConvReluBN((192, 1, 1)),
        ConvReluBN((192, 1, 7), padding=(0, 3)),
        ConvReluBN((192, 7, 1), padding=(3, 0)),
        ConvReluBN((192, 3, 3), stride=2),
    ], [
        layers.MaxPooling((3, 3), stride=(2, 2))
    ]],
    layers.Concatenate(),

    Inception_3(pooling='average'),
    Inception_3(pooling='max'),

    layers.GlobalPooling(),
    layers.Softmax(1000),
)
plots.layer_structure(inception_v3)
