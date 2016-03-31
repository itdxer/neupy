import random

import theano
import numpy as np


__all__ = ('sandbox', 'reproducible')


def sandbox():
    theano.config.linker = 'py'
    theano.config.mode = 'FAST_COMPILE'
    theano.config.optimizer = 'fast_compile'
    theano.config.allow_gc = False


def reproducible(seed=0):
    np.random.seed(seed)
    random.seed(seed)
