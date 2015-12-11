import numpy as np

from neupy.helpers.base import preformat_value


__all__ = (
    'BaseProperty', 'Property',
    'IntProperty', 'NumberProperty', 'ArrayProperty',
    'BoundedProperty', 'ProperFractionProperty',
    'TypedListProperty', 'ChoiceProperty'
)


class BaseProperty(object):
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
                "types: {2}".format(value.__class__.__name__, self.name,
                                    availabe_types)
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
    """ Simple and flexible class that help build properties with
    specified type.
    """
    def __init__(self, expected_type=object, *args, **kwargs):
        self.expected_type = expected_type
        super(Property, self).__init__(*args, **kwargs)


class IntProperty(BaseProperty):
    expected_type = int


class NumberProperty(BaseProperty):
    expected_type = (float, int)


class ArrayProperty(BaseProperty):
    expected_type = (np.ndarray, np.matrix)


class TypedListProperty(BaseProperty):
    expected_type = (list, tuple, set)

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

            raise TypeError("Valid list element types are: {}"
                            "".format(element_type_name))


class ChoiceProperty(BaseProperty):
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
        if value not in self.choices:
            possible_choices = ", ".join(self.choices.keys())
            raise ValueError(
                "Wrong value `{0}` for property `{1}`. Available values: "
                "{2}".format(value, self.name, possible_choices)
            )
        return super(ChoiceProperty, self).__set__(instance, value)

    def __get__(self, instance, value):
        if instance is not None:
            choice_key = super(ChoiceProperty, self).__get__(instance, value)
            return self.choices[choice_key]


class BoundedProperty(NumberProperty):
    def __init__(self, minsize=-np.inf, maxsize=np.inf, *args, **kwargs):
        self.minsize = minsize
        self.maxsize = maxsize
        super(BoundedProperty, self).__init__(*args, **kwargs)

    def validate(self, value):
        if not (self.minsize <= value <= self.maxsize):
            raise ValueError("Value `{}` must be between {} and {}"
                             "".format(self.name, self.minsize, self.maxsize))


class ProperFractionProperty(BoundedProperty):
    def __init__(self, *args, **kwargs):
        super(ProperFractionProperty, self).__init__(
            minsize=0, maxsize=1, *args, **kwargs
        )


class NonNegativeIntProperty(BoundedProperty):
    expected_type = int
