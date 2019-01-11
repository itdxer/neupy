from functools import partial

from neupy import algorithms, layers

from helpers import simple_classification
from helpers import compare_networks
from base import BaseTestCase




    def test_momentum_overfit(self):
        self.assertCanNetworkOverfit(
            partial(algorithms.Momentum, step=0.3, verbose=False),
            epochs=1500,
        )

    def test_nesterov_momentum_overfit(self):
        self.assertCanNetworkOverfit(
            partial(
                algorithms.Momentum,
                step=0.3,
                nesterov=True,
                verbose=False,
            ),
            epochs=1500,
        )
