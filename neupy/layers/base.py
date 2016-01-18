from six import with_metaclass

from neupy.core.config import ConfigMeta, BaseConfigurable
from neupy.layers.connections import ChainConnection


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
    """

    def __init__(self, *args, **options):
        super(BaseLayer, self).__init__()

        self.parameters = []

        # Default variables which will change after initialization
        self.relate_to_layer = None
        self.relate_from_layer = None
        self.layer_id = 1

        # If you will set class method function variable, python understend
        # that this is new class method and will call it with `self`
        # first parameter.
        if hasattr(self.__class__, 'activation_function'):
            self.activation_function = self.__class__.activation_function

        BaseConfigurable.__init__(self, **options)

    def relate_to(self, right_layer):
        self.relate_to_layer = right_layer
        right_layer.relate_from_layer = self
