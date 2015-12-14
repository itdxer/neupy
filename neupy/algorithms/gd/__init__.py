__all__ = ('LEARING_RATE_UPDATE', 'WEIGHT_PENALTY', 'optimization_types',
           'StesSelectionBuiltIn')


# Available optimization types
LEARING_RATE_UPDATE = 1
WEIGHT_PENALTY = 2


optimization_types = {
    LEARING_RATE_UPDATE: "Learning rate update",
    WEIGHT_PENALTY: "Weight penalty",
}


class StepSelectionBuiltIn(object):
    supported_optimizations = {
        WEIGHT_PENALTY: "Weight penalty",
    }
