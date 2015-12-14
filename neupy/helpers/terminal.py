__all__ = ('red', 'green', 'gray', 'bold', 'underline')


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
