import os
import sys
import platform


__all__ = ('red', 'green', 'gray', 'white', 'bold', 'underline')


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


def create_style(ansi_code, use_bright_mode=False):
    """ Create style based on ANSI code number.

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


red = create_style(ansi_code=31)
green = create_style(ansi_code=32)
gray = create_style(ansi_code=37)
white = create_style(ansi_code=37)
bold = create_style(ansi_code=1, use_bright_mode=True)
underline = create_style(ansi_code=4, use_bright_mode=True)
