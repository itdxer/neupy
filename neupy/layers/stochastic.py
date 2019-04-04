import tensorflow as tf

from neupy.utils import asfloat
from neupy.exceptions import LayerConnectionError
from neupy.core.properties import (
    ProperFractionProperty,
    NumberProperty,
    TypedListProperty,
)
from .base import Identity


__all__ = ('Dropout', 'GaussianNoise', 'DropBlock')


def bernoulli_sample(mean, shape):
    samples = tf.random_uniform(shape, minval=0, maxval=1, dtype=tf.float32)
    sign_samples = tf.sign(mean - samples)
    return (sign_samples + 1) / 2


class Dropout(Identity):
    """
    Dropout layer. It randomly switches of (multiplies by zero)
    input values, where probability to be switched per each value
    can be controlled with the ``proba`` parameter. For example,
    ``proba=0.2`` will mean that only 20% of the input values will
    be multiplied by 0 and 80% of the will be unchanged.

    It's important to note that output from the dropout is controlled by
    the ``training`` parameter in the ``output`` method. Dropout
    will be applied only in cases when ``training=True`` propagated
    through the network, otherwise it will act as an identity.

    Parameters
    ----------
    proba : float
        Fraction of the input units to drop. Value needs to be
        between ``0`` and ``1``.

    {Identity.name}

    Methods
    -------
    {Identity.Methods}

    Attributes
    ----------
    {Identity.Attributes}

    See Also
    --------
    :layer:`DropBlock` : DropBlock layer.

    References
    ----------
    [1]  Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever,
         Ruslan Salakhutdinov, Dropout: a simple way to prevent neural
         networks from overfitting, 2014.

    Examples
    --------
    >>> from neupy.layers import *
    >>> network = join(
    ...     Input(10),
    ...     Relu(5) >> Dropout(0.5),
    ...     Relu(5) >> Dropout(0.5),
    ...     Sigmoid(1),
    ... )
    >>> network
    (?, 10) -> [... 6 layers ...] -> (?, 1)
    """
    proba = ProperFractionProperty()

    def __init__(self, proba, name=None):
        super(Dropout, self).__init__(name=name)
        self.proba = proba

    def output(self, input_value, training=False):
        if not training:
            return input_value
        return tf.nn.dropout(input_value, keep_prob=(1.0 - self.proba))


class GaussianNoise(Identity):
    """
    Add gaussian noise to the input value. Mean and standard deviation
    of the noise can be controlled from the layers parameters.

    It's important to note that output from the layer is controled by
    the ``training`` parameter in the ``output`` method. Layer
    will be applied only in cases when ``training=True`` propagated
    through the network, otherwise it will act as an identity.

    Parameters
    ----------
    std : float
        Standard deviation of the gaussian noise. Values needs to
        be greater than zero. Defaults to ``1``.

    mean : float
        Mean of the gaussian noise. Defaults to ``0``.

    {Identity.name}

    Methods
    -------
    {Identity.Methods}

    Attributes
    ----------
    {Identity.Attributes}

    Examples
    --------
    >>> from neupy.layers import *
    >>> network = join(
    ...     Input(10),
    ...     Relu(5) >> GaussianNoise(std=0.1),
    ...     Relu(5) >> GaussianNoise(std=0.1),
    ...     Sigmoid(1),
    ... )
    >>> network
    (?, 10) -> [... 6 layers ...] -> (?, 1)
    """
    mean = NumberProperty()
    std = NumberProperty(minval=0)

    def __init__(self, mean=1, std=0, name=None):
        super(GaussianNoise, self).__init__(name=name)
        self.mean = mean
        self.std = std

    def output(self, input_value, training=False):
        if not training:
            return input_value

        noise = tf.random_normal(
            shape=tf.shape(input_value),
            mean=self.mean,
            stddev=self.std)

        return input_value + noise


class DropBlock(Identity):
    """
    DropBlock, a form of structured dropout, where units in a contiguous
    region of a feature map are dropped together.

    Parameters
    ----------
    keep_proba : float
        Fraction of the input units to keep. Value needs to be
        between ``0`` and ``1``.

    block_size : int or tuple
        Size of the block to be dropped. Blocks that have squared shape can
        be specified with a single integer value. For example, `block_size=5`
        the same as `block_size=(5, 5)`.

    {Identity.name}

    Methods
    -------
    {Identity.Methods}

    Attributes
    ----------
    {Identity.Attributes}

    See Also
    --------
    :layer:`Dropout` : Dropout layer.

    References
    ----------
    [1] Golnaz Ghiasi, Tsung-Yi Lin, Quoc V. Le. DropBlock: A regularization
        method for convolutional networks, 2018.

    Examples
    --------
    >>> from neupy.layers import *
    >>> network = join(
    ...     Input((28, 28, 1)),
    ...
    ...     Convolution((3, 3, 16)) >> Relu(),
    ...     DropBlock(keep_proba=0.1, block_size=5),
    ...
    ...     Convolution((3, 3, 32)) >> Relu(),
    ...     DropBlock(keep_proba=0.1, block_size=5),
    ... )
    """
    keep_proba = ProperFractionProperty()
    block_size = TypedListProperty(n_elements=2)

    def __init__(self, keep_proba, block_size, name=None):
        super(DropBlock, self).__init__(name=name)

        if isinstance(block_size, int):
            block_size = (block_size, block_size)

        self.keep_proba = keep_proba
        self.block_size = block_size

    def get_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)

        if input_shape and input_shape.ndims != 4:
            raise LayerConnectionError(
                "DropBlock layer expects input with 4 dimensions, got {} "
                "with shape {}".format(len(input_shape), input_shape))

        return input_shape

    def output(self, input, training=False):
        if not training:
            return input

        input = tf.convert_to_tensor(input, tf.float32)
        input_shape = tf.shape(input)

        block_height, block_width = self.block_size
        height, width = input_shape[1], input_shape[2]

        input_area = asfloat(width * height)
        block_area = asfloat(block_width * block_height)
        area = asfloat((width - block_width + 1) * (height - block_height + 1))

        mask = bernoulli_sample(
            mean=(1. - self.keep_proba) * input_area / (block_area * area),
            shape=[
                input_shape[0],
                height - block_height + 1,
                width - block_width + 1,
                input_shape[3],
            ],
        )

        br_height = (block_height - 1) // 2
        tl_height = (block_height - 1) - br_height

        br_width = (block_width - 1) // 2
        tl_width = (block_width - 1) - br_width

        mask = tf.pad(mask, [
            [0, 0],
            [tl_height, br_height],
            [tl_width, br_width],
            [0, 0],
        ])
        mask = tf.nn.max_pool(
            mask,
            [1, block_height, block_width, 1],
            strides=[1, 1, 1, 1],
            padding='SAME',
        )
        mask = tf.cast(1 - mask, tf.float32)

        feature_normalizer = asfloat(tf.size(mask)) / tf.reduce_sum(mask)
        return tf.multiply(input, mask) * feature_normalizer
