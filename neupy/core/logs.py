# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals

import sys

from .config import Configurable
from .properties import BaseProperty
from . import terminal


__all__ = ('Verbose',)


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

    def __init__(self, enable=True):
        self.enable = enable
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
            self.stdout.write(text)
            self.stdout.write('\n')

    def newline(self):
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

    def __reduce__(self):
        return (self.__class__, (self.enable,))


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
