import tensorflow as tf

from neupy.utils import asfloat, function_name_scope


__all__ = ('step_decay', 'exponential_decay', 'polynomial_decay')


def init_variables(initial_value, name):
    iteration = tf.Variable(asfloat(0), dtype=tf.float32, name='iteration')
    step = tf.Variable(asfloat(initial_value), dtype=tf.float32, name=name)
    return step, iteration


@function_name_scope
def step_decay(initial_value, reduction_freq, name='step'):
    """
    Algorithm minimizes learning step monotonically after
    each iteration.

    .. math::
        \\alpha_{t + 1} = \\frac{\\alpha_{0}}{1 + \\frac{t}{m}}

    where :math:`\\alpha` is a step, :math:`t` is an iteration number
    and :math:`m` is a ``reduction_freq`` parameter.

    Notes
    -----
    Step will be reduced faster when you have smaller training batches.

    Parameters
    ----------
    initial_value : float
        Initial value for the learning rate.

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
    step, iteration = init_variables(initial_value, name)
    reduction_freq = asfloat(reduction_freq)

    step.updates = [
        (step, initial_value / (1 + iteration / reduction_freq)),
        (iteration, iteration + 1),
    ]

    return step


@function_name_scope
def exponential_decay(initial_value, reduction_freq, reduction_rate,
                      staircase=False, name='step'):
    """
    Applies exponential decay to the learning rate. This function is a
    wrapper for the tensorflow's ``exponential_decay`` function.

    .. math::
        \\alpha_{t + 1} = \\alpha_{0} \\cdot d^{\\frac{t}{r}}

    where :math:`\\alpha` is a step, :math:`t` is an iteration number,
    :math:`d` is a ``reduction_freq`` and :math:`r` is a ``reduction_rate``.

    When ``staircase=True`` and the :math:`\\frac{t}{r}` value will be
    rounded.

    Notes
    -----
    Step will be reduced faster when you have smaller training batches.

    Parameters
    ----------
    initial_value : float
        Initial value for the learning rate.

    reduction_freq : int
        Parameter controls step redution frequency. The larger the
        value the slower step parameter decreases.

    reduction_rate : float
        Parameter controls step redution rate. The larger the
        value the slower step parameter decreases.

    staircase : bool
         If ``True`` decay the learning rate at discrete intervals.
         Defaults to ``False``.

    name : str
        Learning rate's variable name. Defaults to ``step``.

    Examples
    --------
    >>> from neupy import algorithms
    >>> from neupy.layers import *
    >>>
    >>> optimizer = algorithms.Momentum(
    ...     Input(5) > Relu(10) > Sigmoid(1),
    ...     step=algorithms.exponential_decay(
    ...         initial_value=0.1,
    ...         reduction_freq=1000,
    ...         reduction_rate=0.95,
    ...     )
    ... )
    """
    step, iteration = init_variables(initial_value, name)
    step_update = tf.train.exponential_decay(
        learning_rate=step,
        global_step=iteration,
        decay_steps=reduction_freq,
        decay_rate=reduction_rate,
        staircas=staircase,
    )

    step.updates = [
        (step, step_update),
        (iteration, iteration + 1),
    ]

    return step


@function_name_scope
def polynomial_decay(initial_value, decay_steps, minstep=0.001, power=1.0,
                     cycle=False, name='step'):
    """
    Applies polynomial decay to the learning rate. This function is a
    wrapper for the tensorflow's ``polynomial_decay`` function.

    Notes
    -----
    Step will be reduced faster when you have smaller training batches.

    Parameters
    ----------
    initial_value : float
       Initial value for the learning rate.

    name : str
       Learning rate's variable name. Defaults to ``step``.

    Examples
    --------
    >>> from neupy import algorithms
    >>> from neupy.layers import *
    >>>
    >>> optimizer = algorithms.Momentum(
    ...     Input(5) > Relu(10) > Sigmoid(1),
    ...     step=algorithms.polynomial_decay(
    ...         initial_value=0.1,
    ...     )
    ... )
    """
    step, iteration = init_variables(initial_value, name)
    step_update = tf.train.polynomial_decay(
        learning_rate=step,
        global_step=iteration,
        decay_steps=decay_steps,
        end_learning_rate=minstep,
        power=power,
        cycle=cycle,
    )

    step.updates = [
        (step, step),
        (iteration, iteration + 1),
    ]

    return step
