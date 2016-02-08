__all__ = ('LEARING_RATE_UPDATE', 'WEIGHT_PENALTY', 'addon_types',
           'StepSelectionBuiltIn')


# Available add-on types
LEARING_RATE_UPDATE = 1
WEIGHT_PENALTY = 2


addon_types = {
    LEARING_RATE_UPDATE: "Learning rate update",
    WEIGHT_PENALTY: "Weight penalty",
}


class StepSelectionBuiltIn(object):
    supported_addons = {
        WEIGHT_PENALTY: "Weight penalty",
    }


class NoStepSelection(StepSelectionBuiltIn):
    def init_properties(self):
        del self.step
        super(NoStepSelection, self).init_properties()
