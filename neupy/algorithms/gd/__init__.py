__all__ = ('SINGLE_STEP_UPDATE', 'MULTIPLE_STEP_UPDATE', 'WEIGHT_PENALTY',
           'addon_types', 'StepSelectionBuiltIn', 'NoMultipleStepSelection')


# Available add-on types
SINGLE_STEP_UPDATE = 1
MULTIPLE_STEP_UPDATE = 2
WEIGHT_PENALTY = 3


addon_types = {
    SINGLE_STEP_UPDATE: "Single-step update",
    MULTIPLE_STEP_UPDATE: "Multi-step update",
    WEIGHT_PENALTY: "Weight penalty",
}


class StepSelectionBuiltIn(object):
    """ Mixin excludes add-ons that modify learning rate.
    """
    supported_addon_types = [WEIGHT_PENALTY]


class NoMultipleStepSelection(object):
    """ Mixin excludes add-ons that use multiple learning rates for
    one neural network.
    """
    supported_addon_types = [SINGLE_STEP_UPDATE, WEIGHT_PENALTY]


class NoStepSelection(StepSelectionBuiltIn):
    """ Mixin that excludes step property from neural network
    class.
    """
    def init_properties(self):
        del self.step
        super(NoStepSelection, self).init_properties()
