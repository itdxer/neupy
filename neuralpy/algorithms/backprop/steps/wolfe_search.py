from scipy.optimize import line_search

from neuralpy.core.properties import (NonNegativeNumberProperty,
                                      BetweenZeroAndOneProperty)
from neuralpy.algorithms.utils import (matrix_list_in_one_vector,
                                       vector_to_list_of_matrix)
from .base import SingleStep


__all__ = ('WolfeSearch',)


class WolfeSearch(SingleStep):
    """ Wolfe line search for the step selection.

    Parameters
    ----------
    maxstep : float
        Maximum step value. Defaults to ``50``.
    c1 : float
        Parameter for Armijo condition rule. Defaults to ``1e-4``.
    c2 : float
        Parameter for curvature condition rule. Defaults to ``0.9``.

    Attributes
    ----------
    {first_step}

    Warns
    -----
    {bp_depending}
    """

    maxstep = NonNegativeNumberProperty(default=50)
    c1 = BetweenZeroAndOneProperty(default=1e-4)
    c2 = BetweenZeroAndOneProperty(default=0.9)

    def set_weights(self, new_weights):
        for layer, new_weight in zip(self.train_layers, new_weights):
            layer.weight = new_weight.copy()

    def check_updates(self, new_step):
        weights = vector_to_list_of_matrix(
            new_step,
            (layer.size for layer in self.train_layers)
        )
        self.set_weights(weights)
        predicted_output = self.predict(self.input_train)
        return self.error(predicted_output, self.target_train)

    def get_gradient_by_weights(self, weights):
        weights = vector_to_list_of_matrix(
            weights,
            (layer.size for layer in self.train_layers)
        )
        self.set_weights(weights)
        gradient = self.get_gradient(self.output_train,
                                     self.target_train)
        return matrix_list_in_one_vector(gradient)

    def update_weights(self, weight_deltas):
        real_weights = [layer.weight for layer in self.train_layers]

        weights_vector = matrix_list_in_one_vector(real_weights)
        gradients_vetor = matrix_list_in_one_vector(self.gradients)

        res = line_search(self.check_updates,
                          self.get_gradient_by_weights,
                          xk=weights_vector,
                          pk=matrix_list_in_one_vector(weight_deltas),
                          gfk=gradients_vetor,
                          amax=self.maxstep,
                          c1=self.c1,
                          c2=self.c2)

        step = (res[0] if res[0] is not None else self.step)
        # SciPy some times ignore `amax` argument and return
        # bigger result
        self.step = min(self.maxstep, step)
        self.set_weights(real_weights)

        return super(WolfeSearch, self).update_weights(weight_deltas)
