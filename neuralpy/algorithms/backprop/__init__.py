__all__ = ('LEARING_RATE_UPDATE', 'WEIGHT_UPDATE', 'optimization_types')


# Available optimization types
LEARING_RATE_UPDATE = 1
WEIGHT_UPDATE = 2


optimization_types = {
    LEARING_RATE_UPDATE: "Learning rate update",
    WEIGHT_UPDATE: "Weight update",
}
