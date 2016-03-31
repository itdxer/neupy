import numpy as np
import matplotlib.pyplot as plt


def draw_countour(xgrid, ygrid, target_function):
    output = np.zeros((xgrid.shape[0], ygrid.shape[0]))

    for i, x in enumerate(xgrid):
        for j, y in enumerate(ygrid):
            output[j, i] = target_function(x, y)

    X, Y = np.meshgrid(xgrid, ygrid)

    plt.contourf(X, Y, output, 20, alpha=1, cmap='Blues')
    plt.colorbar()


def weight_quiver(weights, color='c'):
    plt.quiver(weights[0, :-1],
               weights[1, :-1],
               weights[0, 1:] - weights[0, :-1],
               weights[1, 1:] - weights[1, :-1],
               scale_units='xy', angles='xy', scale=1,
               color=color)
