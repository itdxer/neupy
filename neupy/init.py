import abc

import six
import numpy as np
import tensorflow as tf

from neupy.core.docs import SharedDocsABCMeta


__all__ = ('Initializer', 'Constant', 'Normal', 'Uniform', 'Orthogonal',
           'HeNormal', 'HeUniform', 'XavierNormal', 'XavierUniform')


def identify_fans(shape):
    """
    Identify fans from shape.

    Parameters
    ----------
    shape : tuple or list

    Returns
    -------
    tuple
        Tuple that contains :math:`fan_{in}` and :math:`fan_{out}`.
    """
    n_dimensions = len(shape)

    if n_dimensions == 0:
        raise ValueError("Cannot apply initializer when shape is unknown")

    elif n_dimensions == 1:
        fan_in, fan_out = shape[0], 1

    elif n_dimensions == 2:
        fan_in, fan_out = shape

    else:
        # By default we assume that weights with more than 2 dimensions
        # are generated for convolutional layers.
        receptive_field = np.prod(shape[:-2]).item(0)
        fan_in = shape[-2] * receptive_field
        fan_out = shape[-1] * receptive_field

    return fan_in, fan_out


def classname(instance):
    return instance.__class__.__name__


def set_numpy_seed(seed=None):
    if seed is not None:
        np.random.seed(seed)


class Initializer(six.with_metaclass(SharedDocsABCMeta)):
    """
    Base class for parameter initialization.

    Methods
    -------
    sample(shape, return_array=False)
        Returns tensorflow's tensor or numpy array with specified
        shape. Type of the object depends on the ``return_array`` value.
        Numpy array will be returned when ``return_array=True`` and
        tensor otherwise.
    """
    inherit_method_docs = True

    @abc.abstractmethod
    def sample(self, shape, return_array=False):
        """
        Returns tensorflow's tensor with specified shape.

        Parameters
        ----------
        shape : tuple
            Parameter shape.

        return_array : bool
            Returns numpy's array when equal to ``True``
            and tensorflow's tensor when equal to ``False``.
            Defaults to ``False``.

        Returns
        -------
        array-like or Tensor
        """
        raise NotImplementedError


class Constant(Initializer):
    """
    Initialize parameter that has constant values.

    Parameters
    ----------
    value : float, int
        All parameters in the tensor will be equal to
        this value. Defaults to ``0``.

    Methods
    -------
    {Initializer.Methods}
    """
    def __init__(self, value=0):
        self.value = value

    def sample(self, shape, return_array=False):
        if return_array:
            return np.ones(shape) * self.value

        elif self.value == 0:
            return tf.zeros(shape)

        return tf.ones(shape) * self.value

    def __repr__(self):
        return '{}({})'.format(classname(self), self.value)


class Normal(Initializer):
    """
    Initialize parameter sampling from the normal distribution.

    Parameters
    ----------
    mean : int, float
        Mean of the normal distribution.

    std : int, float
        Standard deviation of the normal distribution.

    seed : None or int
        Random seed. Integer value will make results reproducible.
        Defaults to ``None``.

    Methods
    -------
    {Initializer.Methods}
    """
    def __init__(self, mean=0, std=0.01, seed=None):
        self.mean = mean
        self.std = std
        self.seed = seed

    def sample(self, shape, return_array=False):
        if return_array:
            set_numpy_seed(self.seed)
            return np.random.normal(loc=self.mean, scale=self.std, size=shape)

        return tf.random_normal(
            mean=self.mean, stddev=self.std,
            shape=shape, seed=self.seed)

    def __repr__(self):
        return '{}(mean={}, std={})'.format(
            classname(self), self.mean, self.std)


class Uniform(Initializer):
    """
    Initialize parameter sampling from the uniform
    distribution.

    Parameters
    ----------
    minval : int, float
        Minimum possible value.

    maxval : int, float
        Maximum possible value.

    seed : None or int
        Random seed. Integer value will make results reproducible.
        Defaults to ``None``.

    Methods
    -------
    {Initializer.Methods}
    """
    def __init__(self, minval=0, maxval=1, seed=None):
        self.minval = minval
        self.maxval = maxval
        self.seed = seed

    def sample(self, shape, return_array=False):
        minval, maxval = self.minval, self.maxval

        if return_array:
            set_numpy_seed(self.seed)
            return np.random.random(shape) * (maxval - minval) + minval

        return tf.random_uniform(shape, minval, maxval, seed=self.seed)

    def __repr__(self):
        return '{}({}, {})'.format(
            classname(self), self.minval, self.maxval)


class Orthogonal(Initializer):
    """
    Initialize matrix with orthogonal basis.

    Parameters
    ----------
    scale : int, float
        Scales output matrix by a specified factor.
        Defaults to ``1``.

    seed : None or int
        Random seed. Integer value will make results reproducible.
        Defaults to ``None``.

    Raises
    ------
    ValueError
        In case if tensor shape has more than 2
        dimensions.

    Methods
    -------
    {Initializer.Methods}
    """
    def __init__(self, scale=1.0, seed=None):
        self.scale = scale
        self.seed = seed

    def sample(self, shape, return_array=False):
        ndim = len(shape)

        if ndim not in (1, 2):
            raise ValueError(
                "Shape attribute must have 1 or 2 dimensions. "
                "Found {} dimensions".format(ndim))

        normal = Normal(seed=self.seed)
        rand_matrix = normal.sample(shape, return_array)

        if ndim == 1:
            return rand_matrix

        if return_array:
            u, _, v = np.linalg.svd(rand_matrix, full_matrices=False)
        else:
            _, u, v = tf.linalg.svd(rand_matrix, full_matrices=False)
            v = tf.transpose(v)

        nrows, ncols = shape
        ortho_base = u if nrows > ncols else v
        return self.scale * ortho_base[:nrows, :ncols]

    def __repr__(self):
        return '{}(scale={})'.format(classname(self), self.scale)


class InitializerWithGain(Initializer):
    """
    Initialization class that has gain property

    Parameters
    ----------
    gain : float
        Scales variance of the distribution by this factor. Value ``2``
        is a suitable choice for layers that have Relu non-linearity.
        Defaults to ``1``.

    seed : None or int
        Random seed. Integer value will make results reproducible.
        Defaults to ``None``.
    """
    def __init__(self, gain=1.0, seed=None):
        self.gain = gain
        self.seed = seed
        super(InitializerWithGain, self).__init__()

    def __repr__(self):
        return '{}(gain={})'.format(classname(self), self.gain)


class HeNormal(InitializerWithGain):
    """
    Kaiming He parameter initialization method based on the
    normal distribution.

    Parameters
    ----------
    {InitializerWithGain.Parameters}

    Methods
    -------
    {Initializer.Methods}

    References
    ----------
    [1] Kaiming He, Xiangyu Zhan, Shaoqing Ren, Jian Sun.
        Delving Deep into Rectifiers: Surpassing Human-Level
        Performance on ImageNet Classification, 2015.
    """
    def sample(self, shape, return_array=False):
        fan_in, _ = identify_fans(shape)
        variance = 1. / fan_in
        std = np.sqrt(self.gain * variance)

        normal = Normal(0, std, seed=self.seed)
        return normal.sample(shape, return_array)


class HeUniform(InitializerWithGain):
    """
    Kaiming He parameter initialization method based on the
    uniformal distribution.

    Parameters
    ----------
    {InitializerWithGain.Parameters}

    Methods
    -------
    {Initializer.Methods}

    References
    ----------
    [1] Kaiming He, Xiangyu Zhan, Shaoqing Ren, Jian Sun.
        Delving Deep into Rectifiers: Surpassing Human-Level
        Performance on ImageNet Classification, 2015.
    """
    def sample(self, shape, return_array=False):
        fan_in, _ = identify_fans(shape)
        variance = 3. / fan_in
        abs_max_value = np.sqrt(self.gain * variance)

        uniform = Uniform(
            minval=-abs_max_value,
            maxval=abs_max_value,
            seed=self.seed,
        )
        return uniform.sample(shape, return_array)


class XavierNormal(InitializerWithGain):
    """
    Xavier Glorot parameter initialization method based on
    normal distribution.

    Parameters
    ----------
    {InitializerWithGain.Parameters}

    Methods
    -------
    {Initializer.Methods}

    References
    ----------
    [1] Xavier Glorot, Y Bengio. Understanding the difficulty
        of training deep feedforward neural networks, 2010.
    """
    def sample(self, shape, return_array=False):
        fan_in, fan_out = identify_fans(shape)
        variance = 1. / (fan_in + fan_out)
        std = np.sqrt(self.gain * variance)

        normal = Normal(0, std, seed=self.seed)
        return normal.sample(shape, return_array)


class XavierUniform(InitializerWithGain):
    """
    Xavier Glorot parameter initialization method based
    on uniform distribution.

    Parameters
    ----------
    {InitializerWithGain.Parameters}

    Methods
    -------
    {Initializer.Methods}

    References
    ----------
    [1] Xavier Glorot, Y Bengio. Understanding the difficulty
        of training deep feedforward neural networks, 2010.
    """
    def sample(self, shape, return_array=False):
        fan_in, fan_out = identify_fans(shape)
        variance = 3. / (fan_in + fan_out)
        abs_max_value = np.sqrt(self.gain * variance)

        uniform = Uniform(
            minval=-abs_max_value,
            maxval=abs_max_value,
            seed=self.seed,
        )
        return uniform.sample(shape, return_array)
