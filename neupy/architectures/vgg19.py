from neupy import layers


__all__ = ('vgg19',)


def vgg19():
    return layers.join(
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
        layers.Convolution((256, 3, 3), padding=1, name='conv3_4') > layers.Relu(),
        layers.MaxPooling((2, 2)),

        layers.Convolution((512, 3, 3), padding=1, name='conv4_1') > layers.Relu(),
        layers.Convolution((512, 3, 3), padding=1, name='conv4_2') > layers.Relu(),
        layers.Convolution((512, 3, 3), padding=1, name='conv4_3') > layers.Relu(),
        layers.Convolution((512, 3, 3), padding=1, name='conv4_4') > layers.Relu(),
        layers.MaxPooling((2, 2)),

        layers.Convolution((512, 3, 3), padding=1, name='conv5_1') > layers.Relu(),
        layers.Convolution((512, 3, 3), padding=1, name='conv5_2') > layers.Relu(),
        layers.Convolution((512, 3, 3), padding=1, name='conv5_3') > layers.Relu(),
        layers.Convolution((512, 3, 3), padding=1, name='conv5_4') > layers.Relu(),
        layers.MaxPooling((2, 2)),

        layers.Reshape(),

        layers.Relu(4096, name='dense_1') > layers.Dropout(0.5),
        layers.Relu(4096, name='dense_2') > layers.Dropout(0.5),
        layers.Softmax(1000, name='dense_3'),
    )
