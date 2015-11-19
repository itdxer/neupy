#!/bin/bash -eE
nosetests \
    tests/steps/test_simple_step_min.py\
    tests/steps/test_error_difference_update.py\
    tests/steps/test_search_then_converge.py\
    tests/steps/test_leak_step_adaptation.py\
    tests/core/test_configs.py\
    tests/core/test_layers.py\
    tests/core/test_signals.py\
    tests/core/test_storage.py\
    tests/core/test_properties.py\
    tests/weights/test_weight_decay.py\
    tests/weights/test_weight_elimination.py\
    tests/rbfn/test_pnn.py\
    tests/rbfn/test_grnn.py\
    tests/plots/test_hinton.py\
    tests/plots/test_error_plot.py\
    tests/datasets/test_reber.py\
    tests/backpropagation/test_quickprop.py\
    tests/backpropagation/test_backpropagation.py\
    tests/backpropagation/test_momentum.py\
    tests/memory/test_discrete_hn.py\
    tests/memory/test_bam.py\
    tests/memory/test_cmac.py\
    tests/compatibilities/test_pandas.py
