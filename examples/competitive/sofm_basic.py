import numpy as np
import matplotlib.pyplot as plt

from neupy import algorithms, utils


utils.reproducible()
plt.style.use('ggplot')


X = np.array([
    [0.1961, 0.9806],
    [-0.1961, 0.9806],
    [0.9806, 0.1961],
    [0.9806, -0.1961],
    [-0.5812, -0.8137],
    [-0.8137, -0.5812],
])

sofmnet = algorithms.SOFM(
    n_inputs=2,
    n_outputs=3,

    step=0.5,
    show_epoch=20,
    shuffle_data=True,
    verbose=True,

    learning_radius=0,
    features_grid=(3, 1),
)

plt.plot(X.T[0:1, :], X.T[1:2, :], 'ko')
sofmnet.train(X, epochs=100)

print("> Start plotting")
plt.xlim(-1, 1.2)
plt.ylim(-1, 1.2)

plt.plot(sofmnet.weight[0:1, :], sofmnet.weight[1:2, :], 'bx')
plt.show()

for data in X:
    print(sofmnet.predict(np.reshape(data, (2, 1)).T))
