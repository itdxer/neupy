import numbers
import inspect

import numpy as np
import tensorflow as tf

from neupy import init
from neupy.utils import as_tuple
from neupy.core.docs import SharedDocs


__all__ = (
    'BaseProperty', 'Property', 'ArrayProperty', 'BoundedProperty',
    'ProperFractionProperty', 'NumberProperty', 'IntProperty',
    'TypedListProperty', 'ChoiceProperty', 'WithdrawProperty',
    'ParameterProperty', 'FunctionWithOptionsProperty',
)


number_type = (int, float, np.floating, np.integer)


class BaseProperty(SharedDocs):
    """
    Base class for properties.

    Parameters
    ----------
    default : object
        Default property value. Defaults to ``None``.

    required : bool
        If parameter equal to ``True`` and value is not defined
        after initialization then exception will be triggered.
        Defaults to ``False``.

    allow_none : bool
        When value is equal to ``True`` than ``None`` is a valid
        value for the parameter. Defaults to ``False``.

    Attributes
    ----------
    name : str or None
        Name of the property. ``None`` in case if name
        wasn't specified.

    expected_type : tuple or object
        Expected data types of the property.
    """
    expected_type = object

    def __init__(self, default=None, required=False, allow_none=False):
        self.name = None
        self.default = default
        self.required = required
        self.allow_none = allow_none

        if allow_none:
            self.expected_type = as_tuple(self.expected_type, type(None))

    def __set__(self, instance, value):
        if not self.allow_none or value is not None:
            self.validate(value)

        instance.__dict__[self.name] = value

    def __get__(self, instance, owner):
        if instance is None:
            return

        if self.default is not None and self.name not in instance.__dict__:
            self.__set__(instance, self.default)

        return instance.__dict__.get(self.name, None)

    def validate(self, value):
        """
        Validate properties value

        Parameters
        ----------
        value : object
        """
        if not isinstance(value, self.expected_type):
            availabe_types = as_tuple(self.expected_type)
            availabe_types = ', '.join(t.__name__ for t in availabe_types)
            dtype = value.__class__.__name__

            raise TypeError(
                "Invalid data type `{0}` for `{1}` property. "
                "Expected types: {2}".format(dtype, self.name, availabe_types))

    def __repr__(self):
        classname = self.__class__.__name__

        if self.name is None:
            return '{}()'.format(classname)

        return '{}(name="{}")'.format(classname, self.name)


class WithdrawProperty(object):
    """
    Defines inherited property that needs to be withdrawn.

    Attributes
    ----------
    name : str or None
        Name of the property. ``None`` in case if name
        wasn't specified.
    """
    def __get__(self, instance, owner):
        # Remove itself, to make sure that instance doesn't
        # have reference to this property. Instead user should
        # be able to see default value from the parent classes,
        # but not allowed to assign different value in __init__
        # method.
        #
        # Other part of functionality defined in the
        # ``ConfigMeta`` class.
        del self


class Property(BaseProperty):
    """
    Simple and flexible class that helps identity properties with
    specified type.

    Parameters
    ----------
    expected_type : object
        Valid data types.

    {BaseProperty.Parameters}
    """
    def __init__(self, expected_type=object, *args, **kwargs):
        self.expected_type = expected_type
        super(Property, self).__init__(*args, **kwargs)


class ArrayProperty(BaseProperty):
    """
    Numpy array or matrix property.

    Parameters
    ----------
    {BaseProperty.Parameters}
    """
    expected_type = (np.ndarray, np.matrix)


class TypedListProperty(BaseProperty):
    """
    List property that contains specified element types.

    Parameters
    ----------
    n_elements : int
        Number of elements in the list. The ``None``
        value mean that list can contains any number of
        elements. Defaults to ``None``.

    element_type : object or tuple
        Type of the elements within the list.

    {BaseProperty.Parameters}
    """
    expected_type = (list, tuple)

    def __init__(self, n_elements=None, element_type=int, *args, **kwargs):
        self.n_elements = n_elements
        self.element_type = element_type
        super(TypedListProperty, self).__init__(*args, **kwargs)

    def validate(self, value):
        super(TypedListProperty, self).validate(value)

        if self.n_elements is not None and len(value) != self.n_elements:
            raise ValueError(
                "Expected list with {} variables".format(self.n_elements))

        if not all(isinstance(v, self.element_type) for v in value):
            element_type = as_tuple(self.element_type)
            type_names = (type_.__name__ for type_ in element_type)
            element_type_name = ', '.join(type_names)

            raise TypeError(
                "The `{}` parameter received invalid element types "
                "in list/tuple. Expected element types: {}, Value: {}"
                "".format(self.name, element_type_name, value))


class ChoiceProperty(BaseProperty):
    """
    Property that can have discrete number of properties.

    Parameters
    ----------
    choices : list, tuple or dict
        Stores all possible choices. Defines list of possible
        choices. If value specified as a dictionary than key
        would be just an alias to the expected value.

    {BaseProperty.Parameters}
    """
    choices = {}

    def __init__(self, choices, *args, **kwargs):
        super(ChoiceProperty, self).__init__(*args, **kwargs)
        self.choices = choices

        if isinstance(choices, (list, tuple, set)):
            self.choices = dict(zip(choices, choices))

        if not isinstance(self.choices, dict):
            class_name = self.choices.__class__.__name__

            raise ValueError(
                "Choice properties can be only a `dict`, got "
                "`{0}`".format(class_name))

        if not self.choices:
            raise ValueError(
                "Must be at least one choice in property "
                "`{0}`".format(self.name))

    def __set__(self, instance, value):
        if value in self.choices:
            return super(ChoiceProperty, self).__set__(instance, value)

        possible_choices = ", ".join(self.choices.keys())
        raise ValueError(
            "Wrong value `{0}` for property `{1}`. Available values: "
            "{2}".format(value, self.name, possible_choices)
        )

    def __get__(self, instance, owner):
        if instance is None:
            return

        choice_key = super(ChoiceProperty, self).__get__(instance, owner)
        return self.choices[choice_key]


class BoundedProperty(BaseProperty):
    """
    Number property that have specified numerical bounds.

    Parameters
    ----------
    minval : float
        Minimum possible value for the property.

    maxval : float
        Maximum possible value for the property.

    {BaseProperty.Parameters}
    """

    def __init__(self, minval=-np.inf, maxval=np.inf, *args, **kwargs):
        self.minval = minval
        self.maxval = maxval
        super(BoundedProperty, self).__init__(*args, **kwargs)

    def validate(self, value):
        super(BoundedProperty, self).validate(value)

        if not (self.minval <= value <= self.maxval):
            raise ValueError(
                "Value `{}` should be between {} and {}"
                "".format(self.name, self.minval, self.maxval))


class ProperFractionProperty(BoundedProperty):
    """
    Proper fraction property. Identify all possible numbers
    between zero and one.

    Parameters
    ----------
    {BaseProperty.Parameters}
    """
    expected_type = (float, int)

    def __init__(self, *args, **kwargs):
        super(ProperFractionProperty, self).__init__(
            minval=0, maxval=1, *args, **kwargs)


class NumberProperty(BoundedProperty):
    """
    Float or integer number property.

    Parameters
    ----------
    {BoundedProperty.Parameters}
    """
    expected_type = number_type


class IntProperty(BoundedProperty):
    """
    Integer property.

    Parameters
    ----------
    {BoundedProperty.Parameters}
    """
    expected_type = (numbers.Integral, np.integer)

    def __set__(self, instance, value):
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        super(IntProperty, self).__set__(instance, value)


class ParameterProperty(ArrayProperty):
    """
    In addition to Numpy arrays and matrix property support also
    Tensorfow variables and NeuPy Initializers.

    Parameters
    ----------
    {ArrayProperty.Parameters}
    """
    expected_type = as_tuple(
        np.ndarray,
        number_type,
        init.Initializer,
        tf.Variable,
        tf.Tensor,
    )

    def __set__(self, instance, value):
        if isinstance(value, number_type):
            value = init.Constant(value)
        super(ParameterProperty, self).__set__(instance, value)


class FunctionWithOptionsProperty(ChoiceProperty):
    """
    Property that helps select error function from
    available or define a new one.

    Parameters
    ----------
    {ChoiceProperty.Parameters}
    """
    def __set__(self, instance, value):
        if inspect.isfunction(value):
            return super(ChoiceProperty, self).__set__(instance, value)

        return super(FunctionWithOptionsProperty, self).__set__(
            instance, value)

    def __get__(self, instance, value):
        founded_value = super(ChoiceProperty, self).__get__(instance, value)

        if inspect.isfunction(founded_value):
            return founded_value

        return super(FunctionWithOptionsProperty, self).__get__(
            instance, founded_value)


class ScalarVariableProperty(BaseProperty):
    expected_type = as_tuple(tf.Variable, number_type)
