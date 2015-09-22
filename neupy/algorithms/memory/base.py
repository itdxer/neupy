from numpy import any as np_any

from neupy.core.base import BaseSkeleton
from neupy.core.properties import ChoiceProperty, NonNegativeIntProperty
from neupy.core.config import Configurable


__all__ = ('DiscreteMemory',)


class DiscreteMemory(BaseSkeleton, Configurable):
    """ Base class for discrete memory networks.

    Notes
    -----
    * {discrete_data_note}
    """
    __discrete_data_note = """ Input and output data must contains only \
    binary values.
    """

    __discrete_params = """mode : {'sync', 'async'}
        Indentify pattern recovery mode. ``sync`` mode try recovery a pattern
        using the all input vector. ``async`` mode randomly chose some
        values from the input vector and repeat this procedure the number
        of times a given variable ``n_times``. Defaults to ``sync``.
    n_times : int
        Available only in ``async`` mode. Identify number of random trials.
        Defaults to ``100``.
    """

    shared_docs = {
        'discrete_data_note': __discrete_data_note,
        'discrete_params': __discrete_params
    }

    mode = ChoiceProperty(default='sync', choices=['async', 'sync'])
    n_times = NonNegativeIntProperty(default=100)

    def __init__(self, **options):
        super(DiscreteMemory, self).__init__(**options)
        self.weight = None

        if 'n_times' in options and self.mode != 'async':
            self.logs.warning("You can use `n_times` property only in "
                              "`async` mode.")

    def discrete_validation(self, matrix):
        """ Validate discrete matrix.

        Parameters
        ----------
        matrix : array-like
            Matrix for validation.

        Returns
        -------
        bool
            Got ``True`` all ``matrix`` discrete values are in
            `discrete_values` list and `False` otherwise.
        """
        if np_any((matrix != 0) & (matrix != 1)):
            raise ValueError("This network is descrete. This mean that you "
                             "can use data which contains 0 and 1 values")
