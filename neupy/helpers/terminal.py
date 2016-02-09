import sys
import platform
from functools import wraps


__all__ = ('red', 'green', 'gray', 'bold', 'underline')


def is_color_supported():
    """ Returns ``True`` if the running system's terminal supports
    color, and ``False`` otherwise.

    Notes
    -----
    Code from Djano: https://github.com/django/django/blob/\
    master/django/core/management/color.py

    Returns
    -------
    bool
    """
    supported_platform = (
        platform != 'Pocket PC' and
        (platform != 'win32' or 'ANSICON' in os.environ)
    )

    # isatty is not always implemented, #6223.
    is_a_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    is_support = (supported_platform and is_a_tty)
    return is_support


def only_if_terminal_support(function):
    """ Decorator for functions that add styles to the input string.
    If terminal do not support it, output will be without modification.
    """
    @wraps(function)
    def wrapper(text, *args, **kwargs):
        if is_color_supported():
            return function(text, *args, **kwargs)
        return text
    return wrapper


@only_if_terminal_support
def red(text):
    """ Makes string color red.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    return "\033[91m{}\033[0m".format(text)


@only_if_terminal_support
def green(text):
    """ Makes string color green.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    return "\033[92m{}\033[0m".format(text)


@only_if_terminal_support
def gray(text):
    """ Makes string color gray.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    return "\033[90m{}\033[0m".format(text)


@only_if_terminal_support
def bold(text):
    """ Makes string font bold.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    return "\033[1m{}\033[0;0m".format(text)


@only_if_terminal_support
def underline(text):
    """ Adds underline to string.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    return "\033[4m{}\033[0;0m".format(text)
