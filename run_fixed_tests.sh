#!/bin/bash -eE
echo 'Run tests...'
nosetests \
    -s -v \
    -e ._testing \
    tests/steps/test_simple_step_min.py\
    tests/steps/test_errdiff.py\
    tests/steps/test_search_then_converge.py\
    tests/steps/test_leak_step_adaptation.py\
    \
    tests/core/test_configs.py\
    tests/core/test_layers.py\
    tests/core/test_signals.py\
    tests/core/test_storage.py\
    tests/core/test_properties.py\
    \
    tests/weights/test_weight_decay.py\
    tests/weights/test_weight_elimination.py\
    \
    tests/rbfn/test_pnn.py\
    tests/rbfn/test_grnn.py\
    tests/rbfn/test_rbf_kmeans.py\
    \
    tests/helpers/test_summary_table.py\
    \
    tests/associative/test_oja.py\
    tests/associative/test_hebb.py\
    tests/associative/test_instar.py\
    tests/associative/test_kohonen.py\
    \
    tests/competitive/test_art.py\
    tests/competitive/test_sofm.py\
    \
    tests/plots/test_hinton.py\
    tests/plots/test_error_plot.py\
    \
    tests/datasets/test_reber.py\
    \
    tests/gd/test_quickprop.py\
    tests/gd/test_gd.py\
    tests/gd/test_rprop.py\
    tests/gd/test_gd_minibatch.py\
    tests/gd/test_momentum.py\
    tests/gd/test_hessdiag.py\
    tests/gd/test_hessian.py\
    tests/gd/test_conjgrad.py\
    \
    tests/linear/test_perceptron.py\
    tests/linear/test_lms.py\
    tests/linear/test_modify_relaxation.py\
    \
    tests/memory/test_discrete_hn.py\
    tests/memory/test_bam.py\
    tests/memory/test_cmac.py\
    \
    tests/compatibilities/test_pandas.py\
    tests/compatibilities/test_sklearn_compatibility.py\
    \
    tests/ensemble/test_dan.py\
    \
    tests/network/test_network_properties.py

# Not fixed tests:
# tests/ensemble/test_mixtures_of_experts.py
#
# tests/gd/test_levenberg_marquardt.py
# tests/gd/test_quasi_newton.py
# tests/gd/test_gd_general.pys
#
# tests/steps/test_wolfe_search.py
# tests/steps/test_linear_search.py

echo ''
echo 'PEP8 validation...'
flake8 neupy/ --exclude=neupy/commands/new_project_template/,__init__.py | nl
