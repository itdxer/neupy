Functions
=========

Each function can be defined with its differential.

.. code-block:: python

    >>> from neupy.functions import with_derivative
    >>>
    >>> def qubic_deriv(x):
    ...     return 3 * x ** 2
    ...
    >>> @with_derivative(qubic_deriv)
    ... def qubic(x):
    ...     return x ** 3
    ...
    ...
    >>> qubic(2)
    8
    >>> qubic.deriv(2)
    12

This notation is important for all algorithms where you need to compute the gradient.
Also you can define derivative for derivative.

.. code-block:: python

    >>> from neupy.functions import with_derivative
    >>>
    >>> def qubic_deriv_deriv(x):
    ...     return 6 * x
    ...
    >>> @with_derivative(qubic_deriv_deriv)
    ... def qubic_deriv(x):
    ...     return 3 * x ** 2
    ...
    >>> @with_derivative(qubic_deriv)
    ... def qubic(x):
    ...     return x ** 3
    ...
    >>>
    >>> qubic(4)
    64
    >>> qubic.deriv(4)
    48
    >>> qubic.deriv.deriv(4)
    24
