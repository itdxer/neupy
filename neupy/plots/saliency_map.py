import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from neupy.exceptions import InvalidConnection


__all__ = ('saliency_map',)


def compile_saliency_map(connection):
    """
    Compile Theano function that returns saliency map.

    Parameters
    ----------
    connection : connection
    """
    x = T.tensor4()

    with connection.disable_training_state():
        output = connection.output(x)

    max_output = T.max(output, axis=1)
    saliency = T.grad(T.sum(max_output), x)

    return theano.function([x], saliency)


def saliency_map(connection, image, mode='heatmap', sigma=8,
                 ax=None, show=True):
    """
    Saliency Map plot.

    Parameters
    ----------
    connection : network, connection
        Network based on which will be computed saliency map.

    image : 3D or 4D array-like tensor
        Image based on which will be computed saliency map.

    mode : {``raw``, ``heatmap``}
        - ``raw``
          Visualize raw gradient. White color on the plot
          defines high gradient values.

        - ``heatmap``
          Applies gaussian filter to the gradient and visualize
          as a heatmap plot.

        Defaults to ``heatmap``.

    sigma : float
        Standard deviation for kernel in Gaussian filter.
        It is used only when ``mode='heatmap'``. Defaults to ``8``.

    ax : object or None
        Matplotlib axis object. ``None`` values means that axis equal
        to the current one (the same as ``ax = plt.gca()``).
        Defaults to ``None``.

    show : bool
        If parameter is equal to ``True`` then plot will be
        displayed. Defaults to ``True``.

    Returns
    -------
    object
        Matplotlib axis instance.

    Examples
    --------
    >>> from neupy import layers, plots
    >>>
    >>> network = layers.join(
    ...     layers.Input((3, 28, 28)),
    ...     layers.Convolution((32, 3, 3)) > layers.Relu(),
    ...     layers.Reshape(),
    ...     layers.Softmax(10),
    ... )
    >>>
    >>> dog_image = load_dog_image()
    >>> plots.saliency_map(network, dog_image)
    """
    if image.ndim not in (3, 4):
        raise ValueError("Invalid image shape. Image expected to be 3D or "
                         "4D, got {}D image".format(image.ndim))

    valid_modes = ('raw', 'heatmap')
    if mode not in valid_modes:
        raise ValueError("{!r} is invalid value for mode argument. Valid "
                         "mode values are: {!r}".format(mode, valid_modes))

    if len(connection.output_layers) != 1:
        raise InvalidConnection(
            "Cannot build saliency map for connection that has more than "
            "one output layer. You need to specify output layer. For "
            "instance, use network.end('layer-name') with specified "
            "output layer name.")

    if ax is None:
        ax = plt.gca()

    # Code here

    if show:
        plt.show()

    return ax
