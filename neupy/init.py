import abc
import math

import six
import numpy as np

from neupy.core.docs import SharedDocsABCMeta


__all__ = ('Initializer', 'Constant', 'Normal', 'Uniform', 'Orthogonal',
           'HeNormal', 'HeUniform', 'XavierNormal', 'XavierUniform')


class UninitializedException(Exception):
    """
    Exception for uninitialized parameters.
    """


def identify_fans(shape):
    """
    Identify fans from shape.

    Parameters
    ----------
    shape : tuple or list
        Matrix shape.

    Returns
    -------
    tuple
        Tuple that contains :math:`fan_{in}` and :math:`fan_{out}`.
    """
    fan_in = shape[0]
    output_feature_shape = shape[1:]

    if output_feature_shape:
        fan_out = np.prod(output_feature_shape).item(0)
    else:
        fan_out = 1

    return fan_in, fan_out


def classname(instance):
    """
    Returns instance class name.

    Parameters
    ----------
    instance : object

    Returns
    -------
    str
    """
    return instance.__class__.__name__


class Initializer(six.with_metaclass(SharedDocsABCMeta)):
    """
    Base class for parameter initialization.

    Methods
    -------
    sample(shape)
        Returns tensor with specified shape.
    """
    inherit_method_docs = True

    @abc.abstractmethod
    def sample(self, shape):
        """
        Returns tensor with specified shape.

        Parameters
        ----------
        shape : tuple
            Parameter shape.

        Returns
        -------
        array-like
        """
        raise NotImplementedError

    def get_value(self):
        """
        This method is the same as ``get_value`` for the Theano
        shared variables. The main point is to be able to
        generate understandable message when user try to get
        value from the uninitialized parameter.
        """
        raise UninitializedException("Cannot get parameter value. "
                                     "Parameter hasn't been initialized yet.")

    def __repr__(self):
        return '{}()'.format(classname(self))


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

    def sample(self, shape):
        return np.ones(shape) * self.value

    def __repr__(self):
        return '{}({})'.format(classname(self), self.value)


class Normal(Initializer):
    """
    Initialize parameter sampling from the normal
    distribution.

    Parameters
    ----------
    mean : int, float
        Mean of the normal distribution.

    std : int, float
        Standard deviation of the normal distribution.

    Methods
    -------
    {Initializer.Methods}
    """
    def __init__(self, mean=0, std=0.01):
        self.mean = mean
        self.std = std

    def sample(self, shape):
        return np.random.normal(loc=self.mean, scale=self.std, size=shape)

    def __repr__(self):
        return '{}(mean={}, std={})'.format(classname(self),
                                            self.mean, self.std)


class Uniform(Initializer):
    """
    Initialize parameter sampling from the uniformal
    distribution.

    Parameters
    ----------
    minval : int, float
        Minimum possible value.

    maxval : int, float
        Maximum possible value.

    Methods
    -------
    {Initializer.Methods}
    """
    def __init__(self, minval=0, maxval=1):
        self.minval = minval
        self.maxval = maxval

    def sample(self, shape):
        minval, maxval = self.minval, self.maxval
        return np.random.random(shape) * (maxval - minval) + minval

    def __repr__(self):
        return '{}({}, {})'.format(classname(self),
                                   self.minval, self.maxval)


class Orthogonal(Initializer):
    """
    Initialize matrix with orthogonal basis.

    Parameters
    ----------
    scale : int, float
        Scales output matrix by a specified factor.
        Defaults to ``1``.

    Raises
    ------
    ValueError
        In case if tensor shape has more than 2
        dimensions.

    Methods
    -------
    {Initializer.Methods}
    """
    def __init__(self, scale=1):
        self.scale = scale

    def sample(self, shape):
        ndim = len(shape)

        if ndim not in (1, 2):
            raise ValueError("Shape attribute must have 1 or 2 dimensions. "
                             "Found {} dimensions".format(ndim))

        rand_matrix = np.random.randn(*shape)

        if ndim == 1:
            return rand_matrix

        nrows, ncols = shape
        u, _, v = np.linalg.svd(rand_matrix, full_matrices=False)
        ortho_base = u if nrows > ncols else v

        return self.scale * ortho_base[:nrows, :ncols]

    def __repr__(self):
        return '{}(scale={})'.format(classname(self), self.scale)


class InitializerWithGain(Initializer):
    """
    Initialization class that has gain property

    Parameters
    ----------
    gain : float or {{``relu``}}
        Multiplies scaling factor by speified gain.
        The ``relu`` values set up gain equal to :math:`\\sqrt{{2}}`.
        Defaults to ``1``.
    """
    def __init__(self, gain=1.0):
        if gain == 'relu':
            gain = math.sqrt(2)

        self.gain = gain
        super(InitializerWithGain, self).__init__()


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
    def sample(self, shape):
        fan_in, _ = identify_fans(shape)
        variance = 2. / fan_in
        std = self.gain * np.sqrt(variance)
        return np.random.normal(loc=0, scale=std, size=shape)


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
    def sample(self, shape):
        fan_in, _ = identify_fans(shape)
        variance = 6. / fan_in
        abs_max_value = self.gain * np.sqrt(variance)

        uniform = Uniform(minval=-abs_max_value, maxval=abs_max_value)
        return uniform.sample(shape)


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
    def sample(self, shape):
        fan_in, fan_out = identify_fans(shape)
        variance = 2. / (fan_in + fan_out)
        std = self.gain * np.sqrt(variance)
        return np.random.normal(loc=0, scale=std, size=shape)


class XavierUniform(InitializerWithGain):
    """
    Xavier Glorot parameter initialization method based
    on uniform distribution.

    Methods
    -------
    {Initializer.Methods}

    References
    ----------
    [1] Xavier Glorot, Y Bengio. Understanding the difficulty
        of training deep feedforward neural networks, 2010.
    """
    def sample(self, shape):
        fan_in, fan_out = identify_fans(shape)
        variance = 6. / (fan_in + fan_out)
        abs_max_value = self.gain * np.sqrt(variance)

        uniform = Uniform(minval=-abs_max_value, maxval=abs_max_value)
        return uniform.sample(shape)
