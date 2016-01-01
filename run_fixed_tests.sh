#!/bin/bash -eE
echo 'Run tests...'
nosetests \
    --with-coverage \
    --cover-package=neupy \
    -s -v \
    -e ._testing \
    tests/algorithms/steps/test_simple_step_min.py\
    tests/algorithms/steps/test_errdiff.py\
    tests/algorithms/steps/test_search_then_converge.py\
    tests/algorithms/steps/test_leak_step_adaptation.py\
    tests/algorithms/steps/test_linear_search.py\
    \
    tests/core/test_configs.py\
    tests/core/test_layers.py\
    tests/core/test_signals.py\
    tests/core/test_storage.py\
    tests/core/test_properties.py\
    tests/core/test_shared_docs.py\
    tests/core/test_utils.py\
    \
    tests/algorithms/weights/test_weight_decay.py\
    tests/algorithms/weights/test_weight_elimination.py\
    \
    tests/algorithms/rbfn/test_pnn.py\
    tests/algorithms/rbfn/test_grnn.py\
    tests/algorithms/rbfn/test_rbf_kmeans.py\
    \
    tests/helpers/test_summary_table.py\
    tests/helpers/test_logging.py\
    \
    tests/algorithms/associative/test_oja.py\
    tests/algorithms/associative/test_hebb.py\
    tests/algorithms/associative/test_instar.py\
    tests/algorithms/associative/test_kohonen.py\
    \
    tests/algorithms/competitive/test_art.py\
    tests/algorithms/competitive/test_sofm.py\
    \
    tests/plots/test_hinton.py\
    tests/plots/test_error_plot.py\
    \
    tests/datasets/test_reber.py\
    \
    tests/algorithms/gd/test_quickprop.py\
    tests/algorithms/gd/test_gd.py\
    tests/algorithms/gd/test_rprop.py\
    tests/algorithms/gd/test_gd_minibatch.py\
    tests/algorithms/gd/test_momentum.py\
    tests/algorithms/gd/test_hessdiag.py\
    tests/algorithms/gd/test_hessian.py\
    tests/algorithms/gd/test_conjgrad.py\
    tests/algorithms/gd/test_levenberg_marquardt.py\
    \
    tests/algorithms/linear/test_perceptron.py\
    tests/algorithms/linear/test_lms.py\
    tests/algorithms/linear/test_modify_relaxation.py\
    \
    tests/algorithms/memory/test_discrete_hn.py\
    tests/algorithms/memory/test_bam.py\
    tests/algorithms/memory/test_cmac.py\
    \
    tests/compatibilities/test_pandas.py\
    tests/compatibilities/test_sklearn_compatibility.py\
    \
    tests/ensemble/test_dan.py\
    tests/ensemble/test_mixtures_of_experts.py\
    \
    tests/network/test_network_properties.py\
    tests/network/test_errors.py

# Not fixed tests:
# tests/gd/test_quasi_newton.py
# tests/gd/test_gd_general.pys

echo ''
echo 'PEP8 validation...'
flake8 neupy/ --exclude=neupy/commands/new_project_template/,__init__.py | nl
