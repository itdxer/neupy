import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def energy(input_vector):
    input_vector = np.array(input_vector)
    X = np.array([[1, -1], [-1, 1]])
    weight = X.T.dot(X) - 2 * np.eye(2)
    return -0.5 * input_vector.dot(weight).dot(input_vector)


fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')

x = y = np.arange(-1.0, 1.0, 0.01)
X, Y = np.meshgrid(x, y)
energies = map(energy, zip(np.ravel(X), np.ravel(Y)))
zs = np.array(list(energies))
Z = zs.reshape(X.shape)

ax.view_init(elev=6, azim=-72)
ax.plot_surface(X, Y, Z, cmap='Reds')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Energy')

plt.show()
