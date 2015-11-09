from six import with_metaclass

from neupy.core.config import ConfigMeta, BaseConfigurable
from neupy.core.properties import IntProperty
from neupy.network.connections import ChainConnection


__all__ = ('BaseLayer',)


class LayerMeta(ConfigMeta):
    def __new__(cls, clsname, bases, attrs):
        if 'activation_function' in attrs:
            # Override `activation_function` in `staticmethod` by default.
            # Python 2 translate any assigned function as class method
            # and try call it with with argument `self` which broke
            # logic. For this reason we try make it static.
            attrs['activation_function'] = staticmethod(
                attrs['activation_function']
            )
        return super(LayerMeta, cls).__new__(cls, clsname, bases, attrs)


class BaseLayer(with_metaclass(LayerMeta, ChainConnection, BaseConfigurable)):
    """ Base class for all layers.

    Parameters
    ----------
    {input_size_param}
    """
    __layer_params = """input_size : int
        Layer input size.
    """
    shared_docs = {'input_size_param': __layer_params}

    input_size = IntProperty()

    def __init__(self, input_size, **options):
        super(BaseLayer, self).__init__()

        self.input_size = input_size

        # Default variables which will change after initialization
        self.relate_to_layer = None
        self.size = None

        # If you will set class method function variable, python understend
        # that this is new class method and will call it with `self`
        # first parameter.
        if hasattr(self.__class__, 'activation_function'):
            self.activation_function = self.__class__.activation_function

        BaseConfigurable.__init__(self, **options)

    def relate_to(self, right_layer):
        self.relate_to_layer = right_layer

    def __repr__(self):
        return '{name}({size})'.format(name=self.__class__.__name__,
                                       size=self.input_size)
