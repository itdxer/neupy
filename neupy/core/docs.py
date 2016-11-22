import re
from inspect import isfunction
from abc import ABCMeta

from six import with_metaclass

from neupy.utils import AttributeKeyDict


__all__ = ("SharedDocsMeta", "SharedDocs", "SharedDocsException",
           "SharedDocsABCMeta", "shared_docs")


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

    if not indent_info:
        return None

    indent, _ = indent_info[0]
    return len(indent)


def iter_doc_parameters(docs):
    """
    Find parameters defined in the documentation.

    Parameters
    ----------
    docs : str

    Yields
    ------
    tuple
        Yields tuple that contain 2 values, namely parameter
        name and full parameter description
    """

    n_indents = find_numpy_doc_indent(docs)
    doc_indent = ' ' * n_indents if n_indents else ''
    parser = re.compile(
        r"(?P<name>\*?\*?\w+)(?P<type>\ *\:\ *[^\n]+)?"
        r"((?P<description>(\n{indent}\ +[^\n]+)|(\n))*)"
        "".format(indent=doc_indent)
    )

    for name, type_, desc, _, _, _ in parser.findall(docs):
        if type_ or name.startswith('*'):
            # In case of *args nad **kwargs we need to clean
            # starts from the beggining
            parameter_name = name.lstrip('*')
            parameter_description = ''.join([name, type_, desc])

            yield parameter_name, parameter_description.rstrip()


def iter_doc_methods(docs):
    """
    Find methods defined in the documentation.

    Parameters
    ----------
    docs : str

    Yields
    ------
    tuple
        Yields tuple that contain 2 values, namely method
        name and full method description
    """

    n_indents = find_numpy_doc_indent(docs)
    doc_indent = ' ' * n_indents if n_indents else ''
    parser = re.compile(
        r"(?P<name>\w+?)(\((.+?)?\))"
        r"((?P<description>\n{indent}\ +[^\n]+)*)"
        "".format(indent=doc_indent)
    )

    for name, func_params, _, desc, _ in parser.findall(docs):
        method_description = ''.join([name, func_params, desc])
        yield name, method_description


def parse_full_section(section_name, docs):
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
    parser = re.compile(r"{}\s+-+\s+(?P<section_text>(.*\n)+?)\s+"
                        # Here we try to find next section title or
                        # the end of the documentation
                        r"([\w\ ]+\n\s+-+\s+|$)"
                        r"".format(section_name))
    parsed_doc_parts = parser.findall(docs)

    if not parsed_doc_parts:
        return None

    section_text_block = parsed_doc_parts[0]
    full_section_text = section_text_block[0]

    # Regexp can catch multiple `\n` symbols at the and of
    # the section. For this reason we need to get rid of them.
    return full_section_text.rstrip()


def parse_variables_from_docs(instances):
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
    # Note: We do not include 'Examples' section because it
    # includes class/function name which will be useless when
    # we inerit documentation for the new object.
    doc_sections = ['Warns', 'Returns', 'Yields', 'Raises', 'See Also',
                    'Parameters', 'Attributes', 'Methods', 'Notes']

    if not instances:
        return variables

    for instance in instances:
        parent_docs = instance.__doc__

        if parent_docs is None:
            continue

        parent_variables = AttributeKeyDict()
        parent_variables.update(iter_doc_parameters(parent_docs))
        parent_variables.update(iter_doc_methods(parent_docs))

        for section_name in doc_sections:
            full_section = parse_full_section(section_name, parent_docs)

            if full_section is not None:
                parent_variables[section_name] = full_section

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
        variables = parse_variables_from_docs(parent_instances)
        return instance_docs.format(**variables)

    except Exception as exception:
        exception_classname = exception.__class__.__name__
        raise SharedDocsException(
            "Can't format documentation for `{}` object. "
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

    Attributes
    ----------
    inherit_method_docs : bool
        ``True`` means that methods that doesn't have
        documentation will be inherited from the parent
        methods. ``False`` will disable this option for
        the specified class. Defaults to ``True``.
    """
    inherit_method_docs = True


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
