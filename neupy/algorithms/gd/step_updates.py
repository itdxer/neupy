import tensorflow as tf

from neupy.utils import asfloat, function_name_scope


__all__ = ('step_decay',)


@function_name_scope
def step_decay(initial_value, reduction_freq, name='step'):
    """
    Algorithm minimizes learning step monotonically after
    each iteration.

    .. math::
        \\alpha_{{t + 1}} = \\frac{{\\alpha_{{0}}}}\
            {{1 + \\frac{{t}}{{m}}}}

    where :math:`\\alpha` is a step, :math:`t` is an iteration number
    and :math:`m` is a ``reduction_freq`` parameter.

    Notes
    -----
    Step will be reduced faster when you have smaller training batches.

    Parameters
    ----------
    initial_value : float
        Initial learning rate.

    reduction_freq : int
        Parameter controls step redution frequency. The larger the
        value the slower step parameter decreases.

        For instance, if ``reduction_freq=100``
        and ``step=0.12`` then after ``100`` iterations ``step`` is
        going to be equal to ``0.06`` (which is ``0.12 / 2``),
        after ``200`` iterations ``step`` is going to be equal to
        ``0.04`` (which is ``0.12 / 3``) and so on.

    name : str
        Learning rate's variable name. Defaults to ``step``.

    Examples
    --------
    >>> from neupy import algorithms
    >>> from neupy.layers import *
    >>>
    >>> optimizer = algorithms.Momentum(
    ...     Input(5) > Relu(10) > Sigmoid(1),
    ...     step=algorithms.step_decay(
    ...         initial_value=0.1,
    ...         reduction_freq=100,
    ...     )
    ... )
    """
    iteration = tf.Variable(asfloat(0), dtype=tf.float32, name='iteration')
    step = tf.Variable(asfloat(initial_value), dtype=tf.float32, name=name)
    reduction_freq = asfloat(reduction_freq)

    step.updates = [
        (step, initial_value / (1 + iteration / reduction_freq)),
        (iteration, iteration + 1),
    ]

    return step
