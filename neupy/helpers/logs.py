from __future__ import print_function

import os
import sys
import importlib
from contextlib import contextmanager

from neupy.core.config import Configurable
from neupy.core.properties import BaseProperty
from neupy.helpers import Progressbar
from . import terminal


__all__ = ('Verbose',)


def terminal_echo(enabled, file_descriptor=sys.stdin):
    """
    Enable/disable echo in the terminal.

    Parameters
    ----------
    enabled : bool
        The `False` value means that you willn't be able to make
        input in terminal.
    file_descriptor : object
        File descriptor that you would like to enable or disable.
        Defaults to ``sys.stdin``.
    """
    is_windows = (os.name == 'nt')

    if is_windows:
        return

    termios = importlib.import_module('termios')

    try:
        attributes = termios.tcgetattr(file_descriptor)
        lflag = attributes[3]

        if enabled:
            lflag |= termios.ECHO
        else:
            lflag &= ~termios.ECHO

        attributes[3] = lflag
        termios.tcsetattr(file_descriptor, termios.TCSANOW, attributes)

    except Exception:
        # This feature is not very important, but any error in
        # this function can crash program, so it's easier to
        # ignore it in case if something went wrong
        pass


class TerminalLogger(object):
    """
    Customized logging class that replace standard logging
    functionality.

    Attributes
    ----------
    enable : bool
        Enable/disable logging output. Defaults to ``True``.
    template : str
        Terminal output message template. Defaults
        to ``"[{name}] {text}"``.
    stdout : object
        Writes output in terminal. Defaults to ``sys.stdout``.
    """

    colors = {
        'gray': terminal.gray,
        'green': terminal.green,
        'red': terminal.red,
        'white': terminal.white,
    }

    def __init__(self):
        self.enable = True
        self.template = "[{tag}] {text}"
        self.stdout = sys.stdout

    def write(self, text):
        """
        Method writes text in terminal if logging is enable.

        Parameters
        ----------
        text : str
        """
        if self.enable:
            self.stdout.write(str(text) + '\n')

    def newline(self):
        """
        Just writes an empty line.
        """
        self.write('\r')

    def message(self, tag, text, color='green'):
        """
        Methods writes message in terminal using specific template.
        Each row should have tag and text. Tag identifies message
        category and text information reletad to this category.

        Parameters
        ----------
        name : str
        text : str
        color : {{'green', 'gray', 'red', 'white'}}
            Property that color text defined as ``tag`` parameter.
            Defaults to ``green``.
        """
        if color not in self.colors:
            available_colors = ', '.join(self.colors.keys())
            raise ValueError("Invalid color `{}`. Available colors: {}"
                             "".format(color, available_colors))

        colorizer = self.colors[color]
        formated_tag = colorizer(tag.upper())
        message = self.template.format(tag=formated_tag, text=text)
        self.write(message)

    def title(self, text):
        """
        Method write text as a title message. Text will be displayed
        using bold and underline text styles. Also there will be empty
        lines before and after the message.

        Parameters
        ----------
        text : str
        """
        bold_text = terminal.bold(text)
        message = "\n{text}\n".format(text=terminal.underline(bold_text))
        self.write(message)

    def error(self, text):
        """
        Method writes messages that related to error type.
        Text will be displayed as message with ``tag`` parameter equal
        to ``'ERROR'``. Color will be red.

        Parameters
        ----------
        text : str
        """
        self.message('ERROR', text, color='red')

    def warning(self, text):
        """
        Method writes messages that related to warning type.
        Text will be displayed as message with ``tag`` parameter equal
        to ``'WARN'``. Color will be red.

        Parameters
        ----------
        text : str
        """
        self.message('WARN', text, color='red')

    def progressbar(self, iterator, *args, **kwargs):
        """
        Make progressbar for specific iteration if logging
        is enable.

        Parameters
        ----------
        iterator : iterable object
        *args
            Arguments for ``Progressbar`` class.
        **kwargs
            Key defined arguments for ``Progressbar`` class.
        """
        if self.enable:
            return Progressbar(iterator, *args, **kwargs)
        return iterator

    @contextmanager
    def disable_user_input(self):
        """
        Context manager helps ignore user input in
        terminal for UNIX.
        """

        if self.enable:
            try:
                terminal_echo(enabled=False)
                yield
            finally:
                terminal_echo(enabled=True)
        else:
            yield


class VerboseProperty(BaseProperty):
    """
    Property that synchronize updates with ``enable`` attribute in
    logging instance.

    Parameters
    ----------
    {BaseProperty.default}
    {BaseProperty.required}
    """
    expected_type = bool

    def __set__(self, instance, value):
        instance.logs.enable = value
        return super(VerboseProperty, self).__set__(instance, value)


class Verbose(Configurable):
    """
    Class that controls NeuPy logging.

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
        self.logs.enable = self.verbose
        super(Verbose, self).__init__(**options)
