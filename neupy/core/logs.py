# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals

import os
import sys

from .config import Configurable
from .properties import BaseProperty


__all__ = ('Verbose',)


def is_color_supported():
    """
    Returns ``True`` if the running system's terminal supports
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
        sys.platform != 'Pocket PC' and
        (sys.platform != 'win32' or 'ANSICON' in os.environ)
    )

    # isatty is not always implemented
    is_a_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    is_support = (supported_platform and is_a_tty)
    return is_support


def create_style(ansi_code, use_bright_mode=False):
    """
    Create style based on ANSI code number.

    Parameters
    ----------
    ansi_code : int
        ANSI style code.

    Returns
    -------
    function
        Function that takes string argument and add ANDI styles
        if its possible.
    """
    def style(text):
        if is_color_supported():
            mode = int(use_bright_mode)
            return "\033[{};{}m{}\033[0m".format(mode, ansi_code, text)
        return text
    return style


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
        'gray': create_style(ansi_code=37),
        'green': create_style(ansi_code=32),
        'red': create_style(ansi_code=31),
        'white': create_style(ansi_code=37),
    }
    styles = {
        'bold': create_style(ansi_code=1, use_bright_mode=True),
        'underline': create_style(ansi_code=4, use_bright_mode=True),
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
        category and text information related to this category.

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
            raise ValueError(
                "Invalid color `{}`. Available colors: {}"
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
        bold = self.styles['bold']
        underline = self.styles['underline']
        self.write("\n{text}\n".format(text=underline(bold(text))))

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
        Property controls verbose output in terminal. The ``True`` value
        enables informative output in the terminal and ``False`` - disable
        it. Defaults to ``False``.

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
