import numpy as np

from neupy.core.properties import ChoiceProperty, IntProperty
from neupy.core.config import Configurable
from neupy.algorithms.base import BaseSkeleton


__all__ = ('DiscreteMemory',)


class DiscreteMemory(BaseSkeleton, Configurable):
    """
    Base class for discrete memory networks.

    Notes
    -----
    - Input and output vectors should contain only binary values.

    Parameters
    ----------
    mode : {{``sync``, ``async``}}
        Specifies pattern recovery mode.

        - ``sync`` mode tries to recover pattern using all
          values from the input vector.

        - ``async`` mode choose randomly some values from the
          input vector and iteratively repeat this procedure.
          Number of iterations defines by the ``n_times``
          parameter.

        Defaults to ``sync``.

    n_times : int
        Available only in ``async`` mode. Identify number
        of random trials. Defaults to ``100``.

    {Verbose.verbose}
    """
    mode = ChoiceProperty(choices=['async', 'sync'])
    n_times = IntProperty(minval=1)

    def __init__(self, mode='sync', n_times=100, verbose=False):
        self.mode = mode
        self.n_times = n_times
        self.weight = None
        super(DiscreteMemory, self).__init__(verbose=verbose)

    def discrete_validation(self, matrix):
        """
        Validate discrete matrix.

        Parameters
        ----------
        matrix : array-like
            Matrix for validation.
        """
        if np.any(~np.isin(matrix, [0, 1])):
            raise ValueError(
                "This network expects only discrete inputs. It mean that "
                "it's possible to can use only matrices with binary values "
                "(0 and 1).")
