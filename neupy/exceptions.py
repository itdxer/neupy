__all__ = (
    'LayerConnectionError', 'InvalidConnection', 'NotTrained',
    'StopTraining', 'WeightInitializationError', 'PropagationError',
)


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


class WeightInitializationError(Exception):
    """
    In case if there is some problem with weight initialization
    """


class PropagationError(Exception):
    """
    Error propagation triggers when input cannot be propagated
    through the layer.
    """
