import re
from abc import ABCMeta
from inspect import isfunction

from six import with_metaclass

from neupy.utils import AttributeKeyDict


__all__ = ("SharedDocsMeta", "SharedDocs", "SharedDocsException",
           "SharedDocsABCMeta", 'shared_docs')


def merge_dicts(left_dict, right_dict):
    """
    Merge two dictionaries in one.

    Parameters
    ----------
    left_dict : dict
    right_dict : dict

    Returns
    -------
    dict
    """
    return dict(left_dict, **right_dict)


def find_numpy_doc_indent(docs):
    """
    Find indent for Numpy styled documentation and return
    number of shifts inside of it.

    Parameters
    ----------
    docs : str

    Returns
    -------
    int or None
        Returns number of indentations in documentation. If
        it doesn't identify indentation function output will
        be ``None`` value.
    """

    indent_detector = re.compile(r"(?P<indent>\ *)(?P<dashes>-{3,})")
    indent_info = indent_detector.findall(docs)

    if indent_info:
        indent, _ = indent_info[0]
        return len(indent)


def iter_parameters(docs):
    """
    Find parameters defined in the documentation.

    Parameters
    ----------
    docs : str

    Yields
    ------
    tuple
        Yields tuple that contain 3 values. There are: parameter
        name, parameter type and parameter description.
    """

    n_indents = find_numpy_doc_indent(docs)
    doc_indent = ' ' * n_indents if n_indents else ''
    parser = re.compile(
        r"(?P<name>\w+?)\s*\:\s*(?P<type>[^\n]+)"
        r"((?P<description>(\n{indent}\ +[^\n]+)|(\n))*)"
        "".format(indent=doc_indent)
    )

    for name, type_, desc, _, _, _ in parser.findall(docs):
        yield (name, type_, desc)


def iter_methods(docs):
    """
    Find methods defined in the documentation.

    Parameters
    ----------
    docs : str

    Yields
    ------
    tuple
        Yields tuple that contain 3 values. There are: method
        name, method parameters and method description.
    """

    n_indents = find_numpy_doc_indent(docs)
    doc_indent = ' ' * n_indents if n_indents else ''
    parser = re.compile(
        r"(?P<name>\w+?)(\((.+?)?\))"
        r"((?P<description>\n{indent}\ +[^\n]+)*)"
        "".format(indent=doc_indent)
    )

    for name, func_params, _, desc, _ in parser.findall(docs):
        yield (name, func_params, desc)


def parse_warns(docs):
    """
    Find warning defined in the documentation.

    Parameters
    ----------
    docs : str

    Returns
    -------
    str or None
        Returns warnings from documentation or ``None`` if
        function didn't find it.
    """

    parser = re.compile(r"Warns\s+-+\s+(?P<warntext>(.+\n)+)")
    doc_warns = parser.findall(docs)

    if not doc_warns:
        return None

    doc_warns, _ = doc_warns[0]
    return doc_warns


def parse_variables_from_docs(instance, parent_instances):
    """
    Parse documentation with NumPy style and returns all
    extracted information.

    Parameters
    ----------
    instances : list
        List of objects that has documentations.

    Returns
    -------
    dict
        Variables parsed from the documentations.
    """
    variables = {}

    if not parent_instances:
        return variables

    for instance in parent_instances:
        parent_docs = instance.__doc__

        if parent_docs is None:
            continue

        parent_variables = AttributeKeyDict()

        for name, type_, desc in iter_parameters(parent_docs):
            parameter = "{} : {}{}".format(name, type_, desc.rstrip())
            parent_variables[name] = parameter

        for name, func_params, desc in iter_methods(parent_docs):
            parent_variables[name] = ''.join([name, func_params, desc])

        doc_warns = parse_warns(parent_docs)
        if doc_warns is not None:
            parent_variables['Warns'] = doc_warns

        parent_name = instance.__name__
        variables[parent_name] = parent_variables

    return variables


def format_docs(instance, parent_instances):
    """
    Format instance's documentation.

    Parameters
    ----------
    instance : object
        Any object that has documentation.
    parent_instances : list
        List of object that has documentations. Function will
        extract all information from theirs documentations and
        it will use them to format main instance documentation.

    Returns
    -------
    str
        Formated documentation.

    Raises
    ------
    SharedDocsException
        If function cannot format documentation properly.
    """
    try:
        instance_docs = instance.__doc__
        variables = parse_variables_from_docs(instance, parent_instances)
        instance_new_docs = instance_docs.format(**variables)
        # If we have multiple spaces between words, we need
        # to trim them. For instance:
        # change "hello   world" to "hello world"
        instance_new_docs = re.sub(
            pattern=r'([\S]+)(\ {2,})([\S]+)',
            repl=r'\1 \3',
            string=instance_new_docs
        )
        return instance_new_docs

    except Exception as exception:
        exception_classname = exception.__class__.__name__
        raise SharedDocsException(
            "Can't format documentation for class `{}`. "
            "Catched `{}` exception with message: {}".format(
                instance.__name__,
                exception_classname,
                exception
            )
        )


class SharedDocsException(Exception):
    """
    Exception that help identify problems related to shared
    documentation.
    """


def has_docs(value):
    """ Checks whether object has documentation.

    Parameters
    ----------
    value : object

    Returns
    -------
    bool
        Function returns ``True`` if object has a documentation
        and ``False`` otherwise.
    """
    return value.__doc__ is not None


def inherit_docs_for_methods(class_, attrs):
    """
    Class methods inherit documentation from the parent
    classes in case if methods doesn't have it.

    Parameters
    ----------
    class_ : object
    attrs : dict
        Class attributes.
    """
    for attrname, attrvalue in attrs.items():
        if not isfunction(attrvalue) or has_docs(attrvalue):
            continue

        for parent_class in class_.__mro__:
            if not hasattr(parent_class, attrname):
                continue

            parent_attrvalue = getattr(parent_class, attrname)
            if has_docs(parent_attrvalue):
                attrvalue.__doc__ = parent_attrvalue.__doc__
                break


class SharedDocsMeta(type):
    """
    Meta-class for shared documentation. This class conatains
    main functionality that help inherit parameters and methods
    descriptions from parent classes. This class automaticaly
    format class documentation using basic python format syntax
    for objects.

    Attributes
    ----------
    inherit_method_docs : bool
        ``True`` means that methods that doesn't have
        documentation will be inherited from the parent
        methods. ``False`` will disable this option for
        the specified class. Defaults to ``True``.
    """
    def __new__(cls, clsname, bases, attrs):
        new_class = super(SharedDocsMeta, cls).__new__(cls, clsname,
                                                       bases, attrs)

        if attrs.get('inherit_method_docs', True):
            inherit_docs_for_methods(new_class, attrs)

        if new_class.__doc__ is None:
            return new_class

        class_docs = new_class.__doc__
        n_indents = find_numpy_doc_indent(class_docs)

        if n_indents is not None:
            new_class.__doc__ = format_docs(new_class, new_class.__mro__)

        return new_class


class SharedDocsABCMeta(SharedDocsMeta, ABCMeta):
    """
    Meta-class that combine ``SharedDocsMeta`` and ``ABCMeta``
    meta-classes.
    """


class SharedDocs(with_metaclass(SharedDocsMeta)):
    """
    Main class that provide with shared documentation
    functionality.
    """


def shared_docs(parent_function):
    """
    Decorator shares documentation between functions.

    Parameters
    ----------
    parent_function : object
        Any object that has documentation.
    """
    def decorator(function):
        function.__doc__ = format_docs(function, [parent_function])
        return function
    return decorator
