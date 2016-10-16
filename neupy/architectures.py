from neupy import layers


__all__ = ('VGG16', 'VGG19')


VGG16 = layers.join(
    layers.Input((3, 224, 224)),

    layers.Convolution((64, 3, 3), border_mode=1) > layers.Relu(),
    layers.Convolution((64, 3, 3), border_mode=1) > layers.Relu(),
    layers.MaxPooling((2, 2)),

    layers.Convolution((128, 3, 3), border_mode=1) > layers.Relu(),
    layers.Convolution((128, 3, 3), border_mode=1) > layers.Relu(),
    layers.MaxPooling((2, 2)),

    layers.Convolution((256, 3, 3), border_mode=1) > layers.Relu(),
    layers.Convolution((256, 3, 3), border_mode=1) > layers.Relu(),
    layers.Convolution((256, 3, 3), border_mode=1) > layers.Relu(),
    layers.MaxPooling((2, 2)),

    layers.Convolution((512, 3, 3), border_mode=1) > layers.Relu(),
    layers.Convolution((512, 3, 3), border_mode=1) > layers.Relu(),
    layers.Convolution((512, 3, 3), border_mode=1) > layers.Relu(),
    layers.MaxPooling((2, 2)),

    layers.Convolution((512, 3, 3), border_mode=1) > layers.Relu(),
    layers.Convolution((512, 3, 3), border_mode=1) > layers.Relu(),
    layers.Convolution((512, 3, 3), border_mode=1) > layers.Relu(),
    layers.MaxPooling((2, 2)),

    layers.Reshape(),

    layers.Relu(4096) > layers.Dropout(0.5),
    layers.Relu(4096) > layers.Dropout(0.5),
    layers.Softmax(1000),
)

VGG19 = layers.join(
    layers.Input((3, 224, 224)),

    layers.Convolution((64, 3, 3), border_mode=1) > layers.Relu(),
    layers.Convolution((64, 3, 3), border_mode=1) > layers.Relu(),
    layers.MaxPooling((2, 2)),

    layers.Convolution((128, 3, 3), border_mode=1) > layers.Relu(),
    layers.Convolution((128, 3, 3), border_mode=1) > layers.Relu(),
    layers.MaxPooling((2, 2)),

    layers.Convolution((256, 3, 3), border_mode=1) > layers.Relu(),
    layers.Convolution((256, 3, 3), border_mode=1) > layers.Relu(),
    layers.Convolution((256, 3, 3), border_mode=1) > layers.Relu(),
    layers.Convolution((256, 3, 3), border_mode=1) > layers.Relu(),
    layers.MaxPooling((2, 2)),

    layers.Convolution((512, 3, 3), border_mode=1) > layers.Relu(),
    layers.Convolution((512, 3, 3), border_mode=1) > layers.Relu(),
    layers.Convolution((512, 3, 3), border_mode=1) > layers.Relu(),
    layers.Convolution((512, 3, 3), border_mode=1) > layers.Relu(),
    layers.MaxPooling((2, 2)),

    layers.Convolution((512, 3, 3), border_mode=1) > layers.Relu(),
    layers.Convolution((512, 3, 3), border_mode=1) > layers.Relu(),
    layers.Convolution((512, 3, 3), border_mode=1) > layers.Relu(),
    layers.Convolution((512, 3, 3), border_mode=1) > layers.Relu(),
    layers.MaxPooling((2, 2)),

    layers.Reshape(),

    layers.Relu(4096) > layers.Dropout(0.5),
    layers.Relu(4096) > layers.Dropout(0.5),
    layers.Softmax(1000),
)
