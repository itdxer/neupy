import theano.tensor as T

from neupy import layers


__all__ = ('squeezenet',)


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


def squeezenet():
    """
    SqueezeNet architecture. This network has small number of
    parameters that can be stored as 5Mb file. The accuracy achived
    on ImageNet comparable to the AlexNet.
    """
    return layers.join(
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
