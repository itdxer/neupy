__all__ = ('LayerConnectionError', 'InvalidConnection', 'NotTrained',
           'StopTraining')


class LayerConnectionError(Exception):
    """
    Error class that triggers in case of connection
    issues within layers.
    """


class InvalidConnection(Exception):
    """
    Connection is not suitable for the specified algorithm
    """


class NotTrained(Exception):
    """
    Algorithms hasn't been trained yet.
    """


class StopTraining(Exception):
    """
    Interrupt training procedure.
    """
