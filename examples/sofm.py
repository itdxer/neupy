import numpy as np
import matplotlib.pyplot as plt

from neupy import algorithms, layers


np.random.seed(0)


input_data = np.array([
    [0.1961, 0.9806],
    [-0.1961, 0.9806],
    [0.9806, 0.1961],
    [0.9806, -0.1961],
    [-0.5812, -0.8137],
    [-0.8137, -0.5812],
])


input_layer = layers.LinearLayer(2)
output_layer = layers.CompetitiveOutputLayer(3)

sofmnet = algorithms.SOFM(
    # Connection
    input_layer > output_layer,
    # SOFM settingsp
    learning_radius=0,
    features_grid=(3, 1),
    # Network settings
    use_bias=False,
    step=0.1,
    show_epoch=100,
    shuffle_data=True,
)

plt.plot(input_data.T[0:1, :], input_data.T[1:2, :], 'ko')
sofmnet.train(input_data, epochs=100)

print("> Draw plot")
plt.xlim(-1, 1.2)
plt.ylim(-1, 1.2)

plt.plot(
    sofmnet.input_layer.weight[0:1, :],
    sofmnet.input_layer.weight[1:2, :],
    'bx'
)
plt.show()

for data in input_data:
    print(sofmnet.predict(np.reshape(data, (2, 1)).T))
