import logging
import operator

from neupy.core.config import Configurable
from neupy.core.properties import BoolProperty


__all__ = ('Verbose',)


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__file__)


def red(text):
    return "\033[91m{}\033[0m".format(text)


def green(text):
    return "\033[92m{}\033[0m".format(text)


def gray(text):
    return "\033[90m{}\033[0m".format(text)


def bold(text):
    return "\033[1m{}\033[0;0m".format(text)


def underline(text):
    return "\033[4m{}\033[0;0m".format(text)


class CustomLogger(logging.Logger):
    def _build_log(self, name, text, color=green):
        return "[{name}] {text}".format(name=color(name.upper()), text=text)

    def log(self, name, text, *args, **kwargs):
        message = self._build_log(name, text, color=green)
        if self.propagate:
            print(message)

    def gray_log(self, name, text, *args, **kwargs):
        message = self._build_log(name, text, color=gray)
        if self.propagate:
            print(message)

    def header(self, text, *args, **kwargs):
        message = "\n{text}\n".format(text=underline(bold(text)))
        if self.propagate:
            print(message)

    def simple(self, text):
        if self.propagate:
            print(text)

    def empty(self):
        if self.propagate:
            print("")

    def error(self, text, *args, **kwargs):
        message = self._build_log('ERROR', text, color=red)
        if self.propagate:
            print(message)

    def warning(self, text, *args, **kwargs):
        message = self._build_log('WARN', text, color=red)
        if self.propagate:
            print(message)

    def data(self, text):
        text = text.strip()
        lines = list(map(operator.methodcaller('strip'), text.splitlines()))
        first_line = lines.pop(0)

        self.simple(bold(first_line))

        for line in lines:
            self.simple("  {} {}".format(green('*'), line))


logging.setLoggerClass(CustomLogger)


class VerboseProperty(BoolProperty):
    def __set__(self, instance, value):
        instance.logs.propagate = value
        return super(VerboseProperty, self).__set__(instance, value)


class Verbose(Configurable):
    verbose = VerboseProperty(default=True)

    def __init__(self, **options):
        logger_name = str(id(self))
        self.logs = logging.getLogger(logger_name)
        super(Verbose, self).__init__(**options)
