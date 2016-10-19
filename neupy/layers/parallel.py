from neupy.core.properties import Property
from .utils import join as layers_join
from .base import BaseLayer


__all__ = ('Parallel',)


class Parallel(BaseLayer):
    """
    Propagate input value through the multiple parallel layer
    connections and then combine output result.

    Parameters
    ----------
    connections : list of lists
        List that contains list of layer connections.

    Methods
    -------
    {BaseLayer.Methods}

    Attributes
    ----------
    {BaseLayer.Attributes}

    Examples
    --------
    >>> from neupy import layers
    >>>
    >>> input_layer = layers.Input((3, 8, 8))
    >>> parallel_layer = layers.Parallel(
    ...     [[
    ...         layers.Convolution((3, 5, 5)),
    ...     ], [
    ...         layers.Convolution((10, 3, 3)),
    ...         layers.Convolution((5, 3, 3)),
    ...     ]],
    ...     layers.Concatenate()
    ... ),
    """
    merge_layer = Property(required=True, expected_type=BaseLayer)

    def __init__(self, connections, merge_layer, **options):
        self.connections = connections.copy()
        super(Parallel, self).__init__(merge_layer=merge_layer, **options)

        for i, connection in enumerate(self.connections):
            if isinstance(connection, (list, tuple)):
                connection = layers_join(connection)
                self.connections[i] = connection

            layers_join(connection, self.merge_layer)

    @BaseLayer.input_shape.setter
    def input_shape(self, value):
        input_layers = self.graph.input_layers

        if not input_layers:
            raise TypeError("Cannot assign new shape. Layer doesn't have "
                            "an input in the graph.")

        input_layer = self.graph.input_layers[0]
        for connection in self.connections:
            layers_join(input_layer, connection)

        self.input_shape_ = value

    @property
    def output_shape(self):
        return self.merge_layer.output_shape

    def output(self, input_value):
        outputs = []
        for connection in self.connections:
            output = connection.output(input_value)
            outputs.append(output)

        return self.merge_layer.output(*outputs)
