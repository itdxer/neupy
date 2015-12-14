from __future__ import print_function

from functools import wraps

from neupy.core.config import Configurable
from neupy.core.properties import BaseProperty
from . import terminal


__all__ = ('Verbose',)


def on_active_propagation(method):
    """ Override method and activate it only in case if ``propagate``
    parameter equal to ``True``.
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if self.propagate:
            return method(self, *args, **kwargs)
    return wrapper


class TerminalLogger(object):
    """ Customized logging class that replace standard logging
    functionality.

    Attributes
    ----------
    propagate : bool
        Enable/disable logging output.
    template : str
        Terminal output message template. Defaults to ``"[{name}] {text}"``.
    stdout : object
        Function that catch message. Defaults to ``print``.
    """

    def __init__(self):
        self.propagate = False
        self.template = "[{name}] {text}"
        self.stdout = print

    def write(self, name, text, color=terminal.green):
        formated_name = color(name.upper())
        message = self.template.format(name=formated_name, text=text)
        self.stdout(message)

    @on_active_propagation
    def log(self, name, text, *args, **kwargs):
        self.write(terminal.green(name), text)

    @on_active_propagation
    def gray_log(self, name, text, *args, **kwargs):
        self.write(terminal.gray(name), text)

    @on_active_propagation
    def header(self, text, *args, **kwargs):
        bold_text = terminal.bold(text)
        message = "\n{text}\n".format(text=terminal.underline(bold_text))
        self.stdout(message)

    @on_active_propagation
    def simple(self, text):
        self.stdout(text)

    @on_active_propagation
    def empty(self):
        self.stdout("")

    @on_active_propagation
    def error(self, text, *args, **kwargs):
        self.write(terminal.red('ERROR'), text)

    @on_active_propagation
    def warning(self, text, *args, **kwargs):
        self.write(terminal.red('WARN'), text)


class VerboseProperty(BaseProperty):
    """ Property that synchronize updates with ``propagate`` attribute in
    logging instance.

    Parameters
    ----------
    {BaseProperty.default}
    {BaseProperty.required}
    """
    expected_type = bool

    def __set__(self, instance, value):
        instance.logs.propagate = value
        return super(VerboseProperty, self).__set__(instance, value)


class Verbose(Configurable):
    """ Class that help use and control NeuPy logging.

    Parameters
    ----------
    verbose : bool
        Property controls verbose output interminal. ``True`` enables
        informative output in the terminal and ``False`` -
        disable it. Defaults to ``False``.

    Attributes
    ----------
    logs : TerminalLogger
        ``TerminalLogger`` instance.
    """
    verbose = VerboseProperty(default=False)

    def __init__(self, **options):
        self.logs = TerminalLogger()
        self.logs.propagate = self.verbose
        super(Verbose, self).__init__(**options)
