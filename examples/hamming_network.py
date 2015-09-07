import numpy as np

from neupy import layers, algorithms


input_data = np.array([
    [-0.1961, 0.9806],
])


input_size, output_size = (2, 3)
input_layer = layers.StepLayer(
    input_size,
    weight=np.array([
        [0.7071, -0.7071],
        [0.7071, 0.7071],
        [-1, 0],
    ]).T,
    function_coef={'lower_value': 0, 'upper_value': 1}
)
output_layer = layers.CompetitiveOutputLayer(output_size)

hamming_network = algorithms.Instar(
    input_layer > output_layer,
    use_bias=False,
    step=0.5,
    n_unconditioned=0
)

with hamming_network as hn:
    hn.train(input_data, epochs=1)
    print(hn.input_layer.weight)
