from functools import wraps, partial

import tensorflow as tf


def regularizer(function):
    @wraps(function)
    def wrapper(exclude, **kwargs):
        return partial(function, **kwargs)
    return wrapper

@regularizer
def l2(weight, decay_rate=0.01):
    return decay_rate * sum(weight)

weight = [1, 2, 3]
regilarizer = l2(decay_rate=0.02, exclude=['bias'])
regilarizer(weight)  # no other arguments allowed
