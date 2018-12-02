import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

from neupy.utils import tensorflow_session, as_tuple
from neupy.exceptions import InvalidConnection


__all__ = ('saliency_map', 'saliency_map_graph')


def saliency_map_graph(connection):
    """
    Returns tensorflow variables for saliency map.

    Parameters
    ----------
    connection : connection
    image : ndarray
    """
    session = tensorflow_session()

    if session in saliency_map_graph.cache:
        return saliency_map_graph.cache[session]

    x = tf.placeholder(
        shape=as_tuple(None, connection.input_shape),
        name='saliency-map/input',
        dtype=tf.float32,
    )

    with connection.disable_training_state():
        prediction = connection.output(x)

    output_class = tf.argmax(prediction[0])
    saliency, = tf.gradients(tf.reduce_max(prediction), x)

    # Caching will ensure that we won't build tensorflow graph every time
    # we generate
    saliency_map_graph.cache[session] = x, saliency, output_class
    return x, saliency, output_class


saliency_map_graph.cache = {}


def saliency_map(connection, image, mode='heatmap', sigma=8,
                 ax=None, show=True, **kwargs):
    """
    Saliency Map plot.

    Parameters
    ----------
    connection : network, connection
        Network based on which will be computed saliency map.

    image : 3D array-like tensor
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
        to the current axes instance (the same as ``ax = plt.gca()``).
        Defaults to ``None``.

    show : bool
        If parameter is equal to ``True`` then plot will be
        displayed. Defaults to ``True``.

    **kwargs
        Arguments for ``plt.imshow`` function.

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
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)

    if image.ndim != 4:
        raise ValueError("Invalid image shape. Image expected to be 3D, "
                         "got {}D image".format(image.ndim))

    valid_modes = ('raw', 'heatmap')
    if mode not in valid_modes:
        raise ValueError("{!r} is invalid value for mode argument. Valid "
                         "mode values are: {!r}".format(mode, valid_modes))

    if len(connection.output_layers) != 1:
        raise InvalidConnection(
            "Cannot build saliency map for connection that has more than "
            "one output layer. Output layer can be specified explicitly. "
            "For instance, use network.end('layer-name') with specified "
            "output layer name.")

    if len(connection.input_layers) != 1:
        raise InvalidConnection(
            "Cannot build saliency map for connection that has more than "
            "one input layer. Output layer can be specified explicitly. "
            "For instance, use network.start('layer-name') with specified "
            "output layer name.")

    if len(connection.input_shape) != 3:
        raise InvalidConnection("Input layer has invalid input shape")

    if ax is None:
        ax = plt.gca()

    x, saliency, output_class = saliency_map_graph(connection)

    session = tensorflow_session()
    saliency, output = session.run(
        [saliency, output_class], feed_dict={x: image})

    saliency = saliency[0].max(axis=-1)

    if mode == 'heatmap':
        saliency = gaussian_filter(saliency, sigma=sigma)

    elif mode == 'raw':
        kwargs.setdefault('cmap', 'gray')

    ax.set_title('Predicted output #{} (0-based indeces)'.format(output))
    ax.imshow(saliency, **kwargs)

    if show:
        plt.show()

    return ax
