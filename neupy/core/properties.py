import numpy as np

from neupy.utils import preformat_value
from neupy.core.docs import SharedDocs


__all__ = ('BaseProperty', 'Property', 'ArrayProperty', 'BoundedProperty',
           'ProperFractionProperty', 'NumberProperty', 'IntProperty',
           'TypedListProperty', 'ChoiceProperty')


class BaseProperty(SharedDocs):
    """ Base class for properties.

    Parameters
    ----------
    default : object
        Default property value. Defaults to ``None``.
    required : bool
        If parameter equal to ``True`` and value undefined after
        initialization class then it will cause an error.
        Defaults to ``False``.
    """
    expected_type = object

    def __init__(self, default=None, required=False):
        self.name = None
        self.default = default
        self.required = required

    def __set__(self, instance, value):
        if not isinstance(value, self.expected_type):
            availabe_types = self.expected_type
            if isinstance(availabe_types, (list, tuple)):
                availabe_types = ', '.join(t.__name__ for t in availabe_types)
            else:
                availabe_types = availabe_types.__name__

            raise TypeError(
                "Invalid data type `{0}` for `{1}` property. Expected "
                "types: {2}".format(
                    value.__class__.__name__,
                    self.name,
                    availabe_types
                )
            )

        self.validate(value)
        instance.__dict__[self.name] = value

    def __get__(self, instance, value):
        if instance is None:
            return

        if self.default is not None and self.name not in instance.__dict__:
            self.__set__(instance, self.default)

        return instance.__dict__.get(self.name, None)

    def __delete__(self, instance):
        name = self.name

        if name in instance.__dict__:
            del instance.__dict__[name]

        if name in instance.options:
            del instance.options[name]

    def __str__(self):
        return str(preformat_value(self.default))

    def __repr__(self):
        return self.__str__()

    def validate(self, value):
        pass


class Property(BaseProperty):
    """ Simple and flexible class that helps indetify properties with
    specified type.

    Parameters
    ----------
    expected_type : object
        Valid data type.
    {BaseProperty.default}
    {BaseProperty.required}
    """
    def __init__(self, expected_type=object, *args, **kwargs):
        self.expected_type = expected_type
        super(Property, self).__init__(*args, **kwargs)


class ArrayProperty(BaseProperty):
    """ Numpy array or matrix property.

    Parameters
    ----------
    {BaseProperty.default}
    {BaseProperty.required}
    """
    expected_type = (np.ndarray, np.matrix)


class TypedListProperty(BaseProperty):
    """ List property that contains specified element types.

    Parameters
    ----------
    n_elements : int
        Indentify fixed number of elements in list. ``None`` value mean
        that list can contains any number of elements. Defaults to ``None``.
    element_type : object or tuple
        There are could be defined valid list elementy type or a bunch
        of them as tuple.
    {BaseProperty.default}
    {BaseProperty.required}
    """
    expected_type = (list, tuple)

    def __init__(self, n_elements=None, element_type=int, *args, **kwargs):
        self.n_elements = n_elements
        self.element_type = element_type
        super(TypedListProperty, self).__init__(*args, **kwargs)

    def validate(self, value):
        super(TypedListProperty, self).validate(value)

        if self.n_elements is not None and len(value) != self.n_elements:
            raise ValueError("Expected list with {} variables"
                             "".format(self.n_elements))

        if not all(isinstance(v, self.element_type) for v in value):
            if isinstance(self.element_type, tuple):
                type_names = (type_.__name__ for type_ in self.element_type)
                element_type_name = ', '.join(type_names)
            else:
                element_type_name = self.element_type.__name__

            raise TypeError("Valid list element types for `{}` are: {}"
                            "".format(self.name, element_type_name))


class ChoiceProperty(BaseProperty):
    """ Property that can have discrete number of properties.

    Parameters
    ----------
    choices : list, tuple or dict
        Identify all posible choices. Dictionary choices ties values
        with some names that can help easily chang options between
        some specific object like functions. List or tuple choices
        do the same as dictionary, but they are useful in case when
        keys and values should be the same.
    {BaseProperty.default}
    {BaseProperty.required}
    """
    choices = {}

    def __init__(self, choices, *args, **kwargs):
        super(ChoiceProperty, self).__init__(*args, **kwargs)
        self.choices = choices

        if isinstance(choices, (list, tuple, set)):
            self.choices = dict(zip(choices, choices))

        if not isinstance(self.choices, dict):
            class_name = self.choices.__class__.__name__
            raise ValueError("Choice properties can be only a `dict`, got "
                             "`{0}`".format(class_name))

        if not self.choices:
            raise ValueError("Must be at least one choice in property "
                             "`{0}`".format(self.name))

    def __set__(self, instance, value):
        if value in self.choices:
            return super(ChoiceProperty, self).__set__(instance, value)

        possible_choices = ", ".join(self.choices.keys())
        raise ValueError(
            "Wrong value `{0}` for property `{1}`. Available values: "
            "{2}".format(value, self.name, possible_choices)
        )

    def __get__(self, instance, value):
        if instance is not None:
            choice_key = super(ChoiceProperty, self).__get__(instance, value)
            return self.choices[choice_key]


class BoundedProperty(BaseProperty):
    """ Number property that have specified numerical bounds.

    Parameters
    ----------
    minval : float
        Minimum possible value for the property.
    maxval : float
        Maximum possible value for the property.
    {BaseProperty.default}
    {BaseProperty.required}
    """

    def __init__(self, minval=-np.inf, maxval=np.inf, *args, **kwargs):
        self.minval = minval
        self.maxval = maxval
        super(BoundedProperty, self).__init__(*args, **kwargs)

    def validate(self, value):
        super(BoundedProperty, self).validate(value)

        if not (self.minval <= value <= self.maxval):
            raise ValueError("Value `{}` should be between {} and {}"
                             "".format(self.name, self.minval, self.maxval))


class ProperFractionProperty(BoundedProperty):
    """ Proper fraction property. Identify all possible numbers
    between zero and one.

    Parameters
    ----------
    {BaseProperty.default}
    {BaseProperty.required}
    """
    expected_type = (float, int)

    def __init__(self, *args, **kwargs):
        super(ProperFractionProperty, self).__init__(minval=0, maxval=1,
                                                     *args, **kwargs)


class NumberProperty(BoundedProperty):
    """ Float or integer number property.

    Parameters
    ----------
    {BoundedProperty.minval}
    {BoundedProperty.maxval}
    {BaseProperty.default}
    {BaseProperty.required}
    """
    expected_type = (float, int)


class IntProperty(BoundedProperty):
    """ Integer property.

    Parameters
    ----------
    {BoundedProperty.minval}
    {BoundedProperty.maxval}
    {BaseProperty.default}
    {BaseProperty.required}
    """
    expected_type = (int, np.integer)
