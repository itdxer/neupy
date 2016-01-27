import theano


__all__ = ('sandbox',)


def sandbox():
    theano.config.linker = "py"
    theano.config.mode = "FAST_COMPILE"
    theano.config.optimizer = "fast_compile"
    theano.config.allow_gc = False
