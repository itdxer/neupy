import tensorflow as tf

from neupy.utils import asfloat, function_name_scope


__all__ = ('step_decay', 'exponential_decay', 'polynomial_decay')


def init_variables(initial_value, iteration=0, name='step'):
    iteration = tf.Variable(
        asfloat(iteration),
        dtype=tf.float32,
        name='iteration',
    )
    step = tf.Variable(
        asfloat(initial_value),
        dtype=tf.float32,
        name=name,
    )
    return step, iteration


@function_name_scope
def step_decay(initial_value, reduction_freq, start_iter=0, name='step'):
    """
    Algorithm minimizes learning step monotonically after
    each iteration.

    .. math::
        \\alpha_{t + 1} = \\frac{\\alpha_{0}}{1 + \\frac{t}{m}}

    where :math:`\\alpha` is a step, :math:`t` is an iteration number
    and :math:`m` is a ``reduction_freq`` parameter.

    .. code-block:: python

        step = initial_value / (1 + current_iteration / reduction_freq)

    Notes
    -----
    Step will be reduced faster when you have smaller training batches.

    Parameters
    ----------
    initial_value : float
        Initial value for the learning rate. It's the learning rate
        returned during the first iteration.

    reduction_freq : int
        Parameter controls step reduction frequency. The larger the
        value the slower step parameter decreases.

        For instance, if ``reduction_freq=100``
        and ``step=0.12`` then after ``100`` iterations ``step`` is
        going to be equal to ``0.06`` (which is ``0.12 / 2``),
        after ``200`` iterations ``step`` is going to be equal to
        ``0.04`` (which is ``0.12 / 3``) and so on.

    start_iter : int
        Start iteration. At has to be equal to ``0`` when network just
        started the training. Defaults to ``0``.

    name : str
        Learning rate's variable name. Defaults to ``step``.

    Examples
    --------
    >>> from neupy import algorithms
    >>> from neupy.layers import *
    >>>
    >>> optimizer = algorithms.Momentum(
    ...     Input(5) >> Relu(10) >> Sigmoid(1),
    ...     step=algorithms.step_decay(
    ...         initial_value=0.1,
    ...         reduction_freq=100,
    ...     )
    ... )
    """
    step, iteration = init_variables(initial_value, start_iter, name)
    reduction_freq = asfloat(reduction_freq)

    step_update = initial_value / (1 + iteration / reduction_freq)
    updated_step = step.assign(step_update)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, updated_step)

    with tf.control_dependencies([updated_step]):
        next_iteration = iteration.assign(iteration + 1)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, next_iteration)

    return step


@function_name_scope
def exponential_decay(initial_value, reduction_freq, reduction_rate,
                      staircase=False, start_iter=0, name='step'):
    """
    Applies exponential decay to the learning rate. This function is a
    wrapper for the tensorflow's ``exponential_decay`` function.

    .. math::
        \\alpha_{t + 1} = \\alpha_{0} \\cdot d^{\\frac{t}{r}}

    where :math:`\\alpha` is a step, :math:`t` is an iteration number,
    :math:`d` is a ``reduction_freq`` and :math:`r` is a ``reduction_rate``.

    .. code-block:: python

        step = initial_value * reduction_rate ^ (
            current_iteration / reduction_freq)

    When ``staircase=True`` the :math:`\\frac{t}{r}` value will be
    rounded.

    .. code-block:: python

        step = initial_value * reduction_rate ^ floor(
            current_iteration / reduction_freq)

    Notes
    -----
    Step will be reduced faster when you have smaller training batches.

    Parameters
    ----------
    initial_value : float
        Initial value for the learning rate.

    reduction_freq : int
        Parameter controls step reduction frequency. The larger the
        value the slower step parameter decreases.

    reduction_rate : float
        Parameter controls step reduction rate. The larger the
        value the slower step parameter decreases.

    staircase : bool
         If ``True`` decay the learning rate at discrete intervals.
         Defaults to ``False``.

    start_iter : int
        Start iteration. At has to be equal to ``0`` when network just
        started the training. Defaults to ``0``.

    name : str
        Learning rate's variable name. Defaults to ``step``.

    Examples
    --------
    >>> from neupy import algorithms
    >>> from neupy.layers import *
    >>>
    >>> optimizer = algorithms.Momentum(
    ...     Input(5) >> Relu(10) >> Sigmoid(1),
    ...     step=algorithms.exponential_decay(
    ...         initial_value=0.1,
    ...         reduction_freq=1000,
    ...         reduction_rate=0.95,
    ...     )
    ... )
    """
    step, iteration = init_variables(initial_value, start_iter, name)
    step_update = tf.train.exponential_decay(
        learning_rate=initial_value,
        global_step=iteration,
        decay_steps=reduction_freq,
        decay_rate=reduction_rate,
        staircase=staircase,
    )

    updated_step = step.assign(step_update)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, updated_step)

    with tf.control_dependencies([updated_step]):
        next_iteration = iteration.assign(iteration + 1)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, next_iteration)

    return step


@function_name_scope
def polynomial_decay(initial_value, decay_iter, minstep=0.001, power=1.0,
                     cycle=False, start_iter=0, name='step'):
    """
    Applies polynomial decay to the learning rate. This function is a
    wrapper for the tensorflow's ``polynomial_decay`` function.

    .. code-block:: python

        iteration = min(current_iteration, decay_iter)
        step = minstep + (
            (initial_value - minstep) *
            (1 - iteration / decay_iter) ^ power
        )

    If cycle is ``True`` then a multiple of ``decay_iter`` is used,
    the first one that is bigger than ``current_iterations``.

    .. code-block:: python

        decay_iter = decay_iter * ceil(current_iteration / decay_iter)
        step = minstep + (
            (initial_value - minstep) *
            (1 - current_iteration / decay_iter) ^ power
        )

    Notes
    -----
    Step will be reduced faster when you have smaller training batches.

    Parameters
    ----------
    initial_value : float
       Initial value for the learning rate.

    decay_iter : int
        When ``cycle=False`` parameter identifies number of iterations
        when ``minstep`` will be reached. When ``cycle=True`` than
        the ``decay_iter`` value will be increased. See code above.

    minstep : float
        Step will never be lower than that minimum possible step,
        specified by this parameter. Defaults to ``0.001``.

    power : float
        The power of the polynomial. Defaults to ``1``.

    cycle : bool
        When value equal to ``True`` than step will be further reduced
        when ``current_iteration > decay_iter``. Defaults to ``False``.

    start_iter : int
        Start iteration. At has to be equal to ``0`` when network just
        started the training. Defaults to ``0``.

    name : str
       Learning rate's variable name. Defaults to ``step``.

    Examples
    --------
    >>> from neupy import algorithms
    >>> from neupy.layers import *
    >>>
    >>> optimizer = algorithms.Momentum(
    ...     Input(5) >> Relu(10) >> Sigmoid(1),
    ...     step=algorithms.polynomial_decay(
    ...         initial_value=0.1,
    ...         decay_iter=1000,
    ...         minstep=0.01,
    ...     )
    ... )
    """
    step, iteration = init_variables(initial_value, start_iter, name)
    step_update = tf.train.polynomial_decay(
        learning_rate=initial_value,
        global_step=iteration,
        decay_steps=decay_iter,
        end_learning_rate=minstep,
        power=power,
        cycle=cycle,
    )

    updated_step = step.assign(step_update)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, updated_step)

    with tf.control_dependencies([updated_step]):
        next_iteration = iteration.assign(iteration + 1)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, next_iteration)

    return step
