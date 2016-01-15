from abc import ABCMeta, abstractmethod, abstractproperty

from six import with_metaclass


__all__ = ('LayerConnection', 'Connection', 'ChainConnection',
           'NetworkConnectionError')


class NetworkConnectionError(Exception):
    pass


class ChainConnection(object):
    def __init__(self):
        self.connection = None

    def __gt__(self, other):
        return LayerConnection(self, other)


class Connection(with_metaclass(ABCMeta, ChainConnection)):
    def __init__(self, left, right):
        super(Connection, self).__init__()

        self.left = left.connection or left
        self.right = right.connection or right

        self.left.connection = self
        self.right.connection = self

        self.connect()

    @abstractmethod
    def connect(self):
        pass

    @abstractproperty
    @property
    def left_layer(self):
        pass

    @abstractproperty
    @property
    def right_layer(self):
        pass

    def __len__(self):
        return len(list(self.__iter__()))

    def __iter__(self):
        if isinstance(self.left, Connection):
            for conn in self.left:
                yield conn
        else:
            yield self.left

        if isinstance(self.right, Connection):
            for conn in self.right:
                yield conn
        else:
            yield self.right

    def __repr__(self):
        layers_reprs = map(repr, self)
        return ' > '.join(layers_reprs)


class LayerConnection(Connection):
    def connect(self):
        self.left_layer.relate_to(self.right_layer)

    @property
    def left_layer(self):
        if isinstance(self.left, Connection):
            return self.left.right_layer
        return self.left

    @property
    def right_layer(self):
        if isinstance(self.right, Connection):
            return self.right.left_layer
        return self.right

    @property
    def input_layer(self):
        if isinstance(self.left, Connection):
            return self.left.input_layer
        return self.left

    @property
    def output_layer(self):
        if isinstance(self.right, Connection):
            return self.right.output_layer
        return self.right
