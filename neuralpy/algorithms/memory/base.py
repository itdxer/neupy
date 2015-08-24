from numpy import any as np_any

from neuralpy.core.base import BaseSkeleton
from neuralpy.core.config import Configurable


__all__ = ('DiscreteMemory',)


class DiscreteMemory(BaseSkeleton, Configurable):
    """ Base class for discrete memory networks.

    Notes
    -----
    * {discrete_data_note}
    """
    __discrete_data_note = """ Input and output data must contains only \
    binary values which are equal to settings in `discrete_values` property.
    """
    shared_docs = {'discrete_data_note': __discrete_data_note}

    def __init__(self, *args, **kwargs):
        self.discrete_values = (0, 1)
        super(DiscreteMemory, self).__init__(*args, **kwargs)

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
        lower_value, upper_value = self.discrete_values

        if np_any((matrix != lower_value) & (matrix != upper_value)):
            raise ValueError("This network is descrete. This mean that you "
                             "can use data which contains {0} and {1} "
                             "values".format(lower_value, upper_value))
