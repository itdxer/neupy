#!/bin/bash -eE
nosetests \
    -e ._testing \
    tests/steps/test_simple_step_min.py\
    tests/steps/test_errdiff.py\
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
    tests/rbfn/test_rbf_kmeans.py\
    tests/helpers/test_summary_table.py\
    tests/associative/test_oja.py\
    tests/competitive/test_art.py\
    tests/plots/test_hinton.py\
    tests/plots/test_error_plot.py\
    tests/datasets/test_reber.py\
    tests/gd/test_quickprop.py\
    tests/gd/test_gd.py\
    tests/gd/test_rprop.py\
    tests/gd/test_gd_minibatch.py\
    tests/gd/test_momentum.py\
    tests/linear/test_perceptron.py\
    tests/linear/test_lms.py\
    tests/linear/test_modify_relaxation.py\
    tests/memory/test_discrete_hn.py\
    tests/memory/test_bam.py\
    tests/memory/test_cmac.py\
    tests/compatibilities/test_pandas.py
