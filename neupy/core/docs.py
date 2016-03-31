import re
from abc import ABCMeta

from six import with_metaclass

from neupy.utils import AttributeKeyDict


__all__ = ("SharedDocsMeta", "SharedDocs", "SharedDocsException",
           "SharedDocsABCMeta")


def merge_dicts(left_dict, right_dict):
    """ Merge two dictionaries in one.

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
    """ Find indent for Numpy styled documentation and return number of
    shifts inside of it.

    Parameters
    ----------
    docs : str

    Returns
    -------
    int or None
        Returns number of indentations in documentation. If it doesn't
        identify indentation function output will be ``None`` value.
    """

    indent_detector = re.compile(r"(?P<indent>\ *)(?P<dashes>-{3,})")
    indent_info = indent_detector.findall(docs)

    if indent_info:
        indent, _ = indent_info[0]
        return len(indent)


def iter_parameters(docs):
    """ Find parameters described in the Numpy style documentation.

    Parameters
    ----------
    docs : str

    Yields
    ------
    tuple
        Yields tuple that contain 3 values. There are: parameter name,
        parameter type and parameter description.
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
    """ Find methods described in the Numpy style documentation.

    Parameters
    ----------
    docs : str

    Yields
    ------
    tuple
        Yields tuple that contain 3 values. There are: method name,
        method parameters and method description.
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
    """ Find warning described in the Numpy style documentation.

    Parameters
    ----------
    docs : str

    Returns
    -------
    str or None
        Returns warnings from documentation or ``None`` if function
        didn't find it.
    """

    parser = re.compile(r"Warns\s+-+\s+(?P<warntext>(.+\n)+)")
    doc_warns = parser.findall(docs)

    if not doc_warns:
        return None

    doc_warns, _ = doc_warns[0]
    return doc_warns


class SharedDocsException(Exception):
    """ Exception that help identify problems related to shared
    documentation.
    """


class SharedDocsMeta(type):
    """ Meta-class for shared documentation. This class conatains main
    functionality that help inherit parameters and methods descriptions
    from parent classes. This class automaticaly format class documentation
    using basic python format syntax for objects.
    """

    def __new__(cls, clsname, bases, attrs):
        new_class = super(SharedDocsMeta, cls).__new__(cls, clsname,
                                                       bases, attrs)
        if new_class.__doc__ is None:
            return new_class

        class_docs = new_class.__doc__
        n_indents = find_numpy_doc_indent(class_docs)

        if n_indents is None:
            return new_class

        parameters = {}
        parent_classes = new_class.__mro__

        for parent_class in parent_classes:
            parent_docs = parent_class.__doc__

            if parent_docs is None:
                continue

            parent_name = parent_class.__name__
            parent_params = parameters[parent_name] = AttributeKeyDict()

            for name, type_, desc in iter_parameters(parent_docs):
                parent_params[name] = "{} : {}{}".format(name, type_, desc)

            for name, func_params, desc in iter_methods(parent_docs):
                parent_params[name] = ''.join([name, func_params, desc])

            doc_warns = parse_warns(parent_docs)
            if doc_warns is not None:
                parent_params['Warns'] = doc_warns

        try:
            new_class.__doc__ = new_class.__doc__.format(**parameters)
        except Exception as exception:
            exception_classname = exception.__class__.__name__
            raise SharedDocsException(
                "Can't format documentation for class `{}`. "
                "Catched `{}` exception with message: {}".format(
                    new_class.__name__,
                    exception_classname,
                    exception
                )
            )

        return new_class


class SharedDocsABCMeta(SharedDocsMeta, ABCMeta):
    """ Meta-class that combine ``SharedDocsMeta`` and ``ABCMeta``
    meta-classes.
    """


class SharedDocs(with_metaclass(SharedDocsMeta)):
    """ Main class that provide with shared documentation functionality.
    """
