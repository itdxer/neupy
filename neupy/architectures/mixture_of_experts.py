import tensorflow as tf

from neupy import layers
from neupy.utils import tf_utils, as_tuple
from neupy.layers.utils import extract_network


__all__ = ('mixture_of_experts',)


def check_if_network_is_valid(network, index):
    if len(network.input_layers) > 1:
        raise ValueError(
            "Each network from the mixture of experts has to process single "
            "input tensor. Network #{} (0-based indeces) has more than one "
            "input layer. Input layers: {}"
            "".format(index, network.output_layers))

    if len(network.output_layers) > 1:
        raise ValueError(
            "Each network from the mixture of experts has to output single "
            "tensor. Network #{} (0-based indeces) has more than one output "
            "layer. Output layers: {}".format(index, network.output_layers))

    if network.input_shape.ndims != 2:
        raise ValueError(
            "Each network from the mixture of experts has to process "
            "only 2-dimensional inputs. Network #{} (0-based indeces) "
            "processes only {}-dimensional inputs. Input layer's shape: {}"
            "".format(index, network.input_shape.ndims, network.input_shape))


def check_if_networks_compatible(networks):
    input_shapes = []
    output_shapes = []

    for i, network in enumerate(networks):
        input_shapes.append(network.input_shape)
        output_shapes.append(network.output_shape)

    for shape in input_shapes:
        if not shape.is_compatible_with(input_shapes[0]):
            raise ValueError(
                "Networks have incompatible input shapes. Shapes: {}"
                "".format(tf_utils.shape_to_tuple(input_shapes)))

    for shape in output_shapes:
        if not shape.is_compatible_with(output_shapes[0]):
            raise ValueError(
                "Networks have incompatible output shapes. Shapes: {}"
                "".format(tf_utils.shape_to_tuple(output_shapes)))


def mixture_of_experts(instances, gating_layer=None):
    """
    Generates mixture of experts architecture from the set of
    networks that has the same input and output shapes.

    Mixture of experts learns to how to mix results from different
    networks in order to get better performances. It adds gating layer
    that using input data tries to figure out which of the networks
    will make better contribution to the final result. The final result
    mixes from all networks using different weights. The higher the weight
    the larger contribution from the individual layer.

    Parameters
    ----------
    instances : list of networks or optimizers
        These networks will be combine into mixture of experts.
        Every network should have single 1D input layer and
        single output layer. Another restriction is that all networks
        should expect the same input and output layers.

    gating_layer : None or layer
        In case if value equal to `None` that the following layer
        will be created.

        .. code-block:: python

            gating_layer = layers.Softmax(len(networks))

        Output from the gating layer should be 1D and equal to
        the number of networks.

    Raises
    ------
    ValueError
        In case if there is some problem with input networks
        or custom gating layer.

    Returns
    -------
    network
        Mixture of experts network that combine all networks into
        single one and adds gating layer to it.

    Examples
    --------
    >>> from neupy import layers, algorithms, architectures
    >>>
    >>> network = architectures.mixture_of_experts([
    ...     layers.join(
    ...         layers.Input(10),
    ...         layers.Relu(5),
    ...     ),
    ...     layers.join(
    ...         layers.Input(10),
    ...         layers.Relu(33),
    ...         layers.Relu(5),
    ...     ),
    ...     layers.join(
    ...         layers.Input(10),
    ...         layers.Relu(12),
    ...         layers.Relu(25),
    ...         layers.Relu(5),
    ...     ),
    ... ])
    >>> network
    10 -> [... 12 layers ...] -> 5
    >>>
    >>> gdnet = algorithms.Momentum(network, step=0.1)
    """
    if not isinstance(instances, (list, tuple)):
        raise ValueError("Networks should be specified as a list")

    networks = []
    for index, instance in enumerate(instances):
        network = extract_network(instance)
        check_if_network_is_valid(network, index)
        networks.append(network)

    check_if_networks_compatible(networks)
    input_shape = tf.TensorShape(None)

    for network in networks:
        input_shape = input_shape.merge_with(network.input_shape)

    n_layers_to_combine = len(networks)
    n_features = input_shape[1].value

    if n_features is None:
        raise ValueError(
            "Cannot create mixture of experts model, because "
            "number of input features is unknown")

    if gating_layer is None:
        gating_layer = layers.Softmax(n_layers_to_combine)

    if not isinstance(gating_layer, layers.BaseLayer):
        raise ValueError(
            "Invalid type for gating layer. Type: {}"
            "".format(type(gating_layer)))

    return layers.join(
        layers.Input(n_features),
        # Note: Gating network should be specified
        # as a first parameter.
        layers.parallel(*as_tuple(gating_layer, networks)),
        layers.GatedAverage(),
    )
