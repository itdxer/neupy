from contextlib import contextmanager


__all__ = ('LayerConnection', 'ChainConnection', 'NetworkConnectionError',
           'LayerConnectionError')


class LayerConnectionError(Exception):
    """
    Error class that triggers in case of connection
    issues within layers.
    """


class NetworkConnectionError(Exception):
    """
    Error class that triggers in case of connection
    within layers in the network
    """


class ChainConnection(object):
    def __init__(self):
        self.connection = None
        self.training_state = True

    def __gt__(self, other):
        return LayerConnection(self, other)

    def __lt__(self, other):
        return LayerConnection(other, self)

    @property
    def input_shape(self):
        raise NotImplementedError

    @property
    def output_shape(self):
        raise NotImplementedError

    def output(self, input_value):
        raise NotImplementedError

    def initialize(self):
        pass

    @contextmanager
    def disable_training_state(self):
        self.training_state = False
        yield
        self.training_state = True


class LayerConnection(ChainConnection):
    """
    Connect to layers or connections together.

    Parameters
    ----------
    left : ChainConnection instance
    right : ChainConnection instance
    """
    def __init__(self, left, right):
        super(LayerConnection, self).__init__()

        self.left = left.connection or left
        self.right = right.connection or right

        input_shape = self.right.input_shape
        output_shape = self.left.output_shape

        if input_shape and output_shape and output_shape != input_shape:
            raise NetworkConnectionError(
                "Cannot connect {} to the {}. Output shape from one "
                "layer is equal to {} and Input shape to the next "
                "one is equal to {}".format(
                    self.left, self.right,
                    output_shape, input_shape,
                )
            )

        self.left.connection = self
        self.right.connection = self

        self.left_layer.relate_to(self.right_layer)

    @property
    def left_layer(self):
        if isinstance(self.left, LayerConnection):
            all_elements = list(self.left)
            last_element = all_elements[-1]
            return last_element
        return self.left

    @property
    def right_layer(self):
        if isinstance(self.right, LayerConnection):
            return next(iter(self.right))
        return self.right

    @property
    def input_layer(self):
        if isinstance(self.left, LayerConnection):
            return self.left.input_layer
        return self.left

    @property
    def output_layer(self):
        if isinstance(self.right, LayerConnection):
            return self.right.output_layer
        return self.right

    @property
    def input_shape(self):
        return self.input_layer.input_shape

    @property
    def output_shape(self):
        return self.output_layer.output_shape

    def initialize(self):
        for layer in self:
            layer.initialize()

    def output(self, input_value):
        output_value = input_value
        for layer in self:
            output_value = layer.output(output_value)
        return output_value

    def __len__(self):
        layers = list(iter(self))
        return len(layers)

    def __iter__(self):
        if isinstance(self.left, LayerConnection):
            for conn in self.left:
                yield conn
        else:
            yield self.left

        if isinstance(self.right, LayerConnection):
            for conn in self.right:
                yield conn
        else:
            yield self.right

    def __repr__(self):
        layers_reprs = map(repr, self)
        return ' > '.join(layers_reprs)
