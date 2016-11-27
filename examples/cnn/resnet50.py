from neupy import layers, plots


def ResidualUnit(n_in_filters, n_out_filters, stride, has_branch=False):
    main_branch = layers.join(
        layers.Convolution((n_in_filters, 1, 1), stride=stride, bias=None),
        layers.BatchNorm(),
        layers.Relu(),

        layers.Convolution((n_in_filters, 3, 3), padding=1, bias=None),
        layers.BatchNorm(),
        layers.Relu(),

        layers.Convolution((n_out_filters, 1, 1), bias=None),
        layers.BatchNorm(),
    )

    residual_branch = []
    if has_branch:
        residual_branch = layers.join(
            layers.Convolution((n_out_filters, 1, 1),
                               stride=stride, bias=None),
            layers.BatchNorm(),
        )

    return layers.join(
        [main_branch, residual_branch],
        layers.Elementwise() > layers.Relu(),
    )


resnet50 = layers.join(
    layers.Input((3, 224, 224)),

    layers.Convolution((64, 7, 7), stride=2, padding=3),
    layers.BatchNorm(),
    layers.Relu(),

    layers.MaxPooling((3, 3), stride=(2, 2), ignore_border=False),

    ResidualUnit(64, 256, stride=1, has_branch=True),
    ResidualUnit(64, 256, stride=1),
    ResidualUnit(64, 256, stride=1),

    ResidualUnit(128, 512, stride=2, has_branch=True),
    ResidualUnit(128, 512, stride=1),
    ResidualUnit(128, 512, stride=1),
    ResidualUnit(128, 512, stride=1),

    ResidualUnit(256, 1024, stride=2, has_branch=True),
    ResidualUnit(256, 1024, stride=1),
    ResidualUnit(256, 1024, stride=1),
    ResidualUnit(256, 1024, stride=1),
    ResidualUnit(256, 1024, stride=1),
    ResidualUnit(256, 1024, stride=1),

    ResidualUnit(512, 2048, stride=2, has_branch=True),
    ResidualUnit(512, 2048, stride=1),
    ResidualUnit(512, 2048, stride=1),

    layers.GlobalPooling(),
    layers.Reshape(),
    layers.Softmax(1000),
)
plots.layer_structure(resnet50)
