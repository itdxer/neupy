from functools import partial

from neupy import algorithms

from data import simple_input_train, simple_target_train
from utils import compare_networks
from base import BaseTestCase


class WolfeSearchTestCase(BaseTestCase):
    def test_wolfe_search(self):
        network_default_error, network_tested_error = compare_networks(
            # Test classes
            algorithms.Backpropagation,
            partial(
                algorithms.Backpropagation,
                optimizations=[algorithms.WolfeSearch],
                maxstep=10,
                c1=1e-5,
                c2=0.95,
            ),
            # Test data
            (simple_input_train, simple_target_train),
            # Network configurations
            connection=(3, 10, 2),
            step=0.2,
            use_raw_predict_at_error=False,
            shuffle_data=True,
            verbose=False,
            # Test configurations
            epochs=50,
            # is_comparison_plot=True
        )
        self.assertGreater(network_default_error, network_tested_error)
