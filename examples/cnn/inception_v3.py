from functools import partial

from neupy import layers, plots


def Conv_Relu_BatchNorm(*conv_args, **conv_kwargs):
    return layers.join(
        layers.Convolution(*conv_args, **conv_kwargs),
        layers.Relu(),
        layers.BatchNorm(epsilon=0.001),
    )


def Inception_1(conv_filters):
    return layers.join(
        [[
            Conv_Relu_BatchNorm((conv_filters[0][0], 1, 1)),
        ], [
            Conv_Relu_BatchNorm((conv_filters[1][0], 1, 1)),
            Conv_Relu_BatchNorm((conv_filters[1][1], 5, 5), padding=2),
        ], [
            Conv_Relu_BatchNorm((conv_filters[2][0], 1, 1)),
            Conv_Relu_BatchNorm((conv_filters[2][1], 3, 3), padding=1),
            Conv_Relu_BatchNorm((conv_filters[2][2], 3, 3), padding=1),
        ], [
            layers.AveragePooling((3, 3), stride=(1, 1), padding=1,
                                  mode='exclude_padding'),
            Conv_Relu_BatchNorm((conv_filters[3][0], 1, 1)),
        ]],
        layers.Concatenate(),
    )


def Inception_2(conv_filters):
    return layers.join(
        [[
            Conv_Relu_BatchNorm((conv_filters[0][0], 1, 1)),
        ], [
            Conv_Relu_BatchNorm((conv_filters[1][0], 1, 1)),
            Conv_Relu_BatchNorm((conv_filters[1][1], 1, 7), padding=(0, 3)),
            Conv_Relu_BatchNorm((conv_filters[1][2], 7, 1), padding=(3, 0)),
        ], [
            Conv_Relu_BatchNorm((conv_filters[2][0], 1, 1)),
            Conv_Relu_BatchNorm((conv_filters[2][1], 7, 1), padding=(3, 0)),
            Conv_Relu_BatchNorm((conv_filters[2][2], 1, 7), padding=(0, 3)),
            Conv_Relu_BatchNorm((conv_filters[2][3], 7, 1), padding=(3, 0)),
            Conv_Relu_BatchNorm((conv_filters[2][4], 1, 7), padding=(0, 3)),
        ], [
            layers.AveragePooling((3, 3), stride=(1, 1), padding=1,
                                  mode='exclude_padding'),
            Conv_Relu_BatchNorm((conv_filters[3][0], 1, 1)),
        ]],
        layers.Concatenate(),
    )


def Inception_3(conv_filters, pooling):
    if pooling not in ('max', 'average'):
        raise ValueError("Invalid pooling option: {}".format(pooling))

    if pooling == 'max':
        Pooling = layers.MaxPooling

    elif pooling == 'average':
        Pooling = partial(layers.AveragePooling, mode='exclude_padding')

    return layers.join(
        [[
            Conv_Relu_BatchNorm((conv_filters[0][0], 1, 1)),
        ], [
            Conv_Relu_BatchNorm((conv_filters[1][0], 1, 1)),
            [[
                Conv_Relu_BatchNorm((conv_filters[1][1], 1, 3),
                                    padding=(0, 1)),
            ], [
                Conv_Relu_BatchNorm((conv_filters[1][2], 3, 1),
                                    padding=(1, 0)),
            ]],
        ], [
            Conv_Relu_BatchNorm((conv_filters[2][0], 1, 1)),
            Conv_Relu_BatchNorm((conv_filters[2][1], 3, 3), padding=1),
            [[
                Conv_Relu_BatchNorm((conv_filters[2][2], 1, 3),
                                    padding=(0, 1)),
            ], [
                Conv_Relu_BatchNorm((conv_filters[2][3], 3, 1),
                                    padding=(1, 0)),
            ]],
        ], [
            Pooling((3, 3), stride=(1, 1), padding=1),
            Conv_Relu_BatchNorm((conv_filters[3][0], 1, 1)),
        ]],
        layers.Concatenate(),
    )


inception_v3 = layers.join(
    layers.Input((3, 299, 299)),

    Conv_Relu_BatchNorm((32, 3, 3), stride=2),
    Conv_Relu_BatchNorm((32, 3, 3)),
    Conv_Relu_BatchNorm((64, 3, 3), padding=1),
    layers.MaxPooling((3, 3), stride=(2, 2)),

    Conv_Relu_BatchNorm((80, 1, 1)),
    Conv_Relu_BatchNorm((192, 3, 3)),
    layers.MaxPooling((3, 3), stride=(2, 2)),

    Inception_1([[64], [48, 64], [64, 96, 96], [32]]),
    Inception_1([[64], [48, 64], [64, 96, 96], [64]]),
    Inception_1([[64], [48, 64], [64, 96, 96], [64]]),

    [[
        Conv_Relu_BatchNorm((384, 3, 3), stride=2),
    ], [
        Conv_Relu_BatchNorm((64, 1, 1)),
        Conv_Relu_BatchNorm((96, 3, 3), padding=1),
        Conv_Relu_BatchNorm((96, 3, 3), stride=2),

    ], [
        layers.MaxPooling((3, 3), stride=(2, 2))
    ]],
    layers.Concatenate(),

    Inception_2([[192], [128, 128, 192], [128, 128, 128, 128, 192], [192]]),
    Inception_2([[192], [160, 160, 192], [160, 160, 160, 160, 192], [192]]),
    Inception_2([[192], [160, 160, 192], [160, 160, 160, 160, 192], [192]]),
    Inception_2([[192], [192, 192, 192], [192, 192, 192, 192, 192], [192]]),

    [[
        Conv_Relu_BatchNorm((192, 1, 1)),
        Conv_Relu_BatchNorm((320, 3, 3), stride=2),
    ], [
        Conv_Relu_BatchNorm((192, 1, 1)),
        Conv_Relu_BatchNorm((192, 1, 7), padding=(0, 3)),
        Conv_Relu_BatchNorm((192, 7, 1), padding=(3, 0)),
        Conv_Relu_BatchNorm((192, 3, 3), stride=2),
    ], [
        layers.MaxPooling((3, 3), stride=(2, 2))
    ]],
    layers.Concatenate(),

    Inception_3([[320], [384, 384, 384], [448, 384, 384, 384], [192]],
                pooling='average'),
    Inception_3([[320], [384, 384, 384], [448, 384, 384, 384], [192]],
                pooling='max'),

    layers.GlobalPooling(),
    layers.Softmax(1000),
)
plots.layer_structure(inception_v3)
