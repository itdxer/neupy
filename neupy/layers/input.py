from neupy.utils import as_tuple
from neupy.core.properties import TypedListProperty
from .base import BaseLayer


__all__ = ('Input',)


class ArrayShapeProperty(TypedListProperty):
    """
    Property that identifies array's shape.

    Parameters
    ----------
    {TypedListProperty.Parameters}
    """
    expected_type = (int, tuple)

    def validate(self, value):
        if not isinstance(value, int):
            super(ArrayShapeProperty, self).validate(value)

        elif value < 1:
            raise ValueError("Integer value is expected to be greater or "
                             "equal to one for the `{}` property, got {}"
                             "".format(self.name, value))


class Input(BaseLayer):
    """
    Input layer defines input's feature shape.

    Parameters
    ----------
    size : int, tuple or None
        Identifies input's feature shape.

    {BaseLayer.Parameters}

    Methods
    -------
    {BaseLayer.Methods}

    Attributes
    ----------
    {BaseLayer.Attributes}

    Examples
    --------
    >>> from neupy import layers
    >>> input_layer = layers.Input(10)
    >>> input_layer
    Input(10)
    """
    size = ArrayShapeProperty(element_type=(int, type(None)))

    def __init__(self, size, **options):
        super(Input, self).__init__(size=size, **options)

        self.input_shape = as_tuple(self.size)
        self.initialize()

    def __repr__(self):
        classname = self.__class__.__name__
        return '{name}({size})'.format(name=classname, size=self.size)
