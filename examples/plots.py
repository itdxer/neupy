import numpy as np
import matplotlib.pyplot as plt


# Got color from page:
# http://matplotlib.org/examples/pylab_examples/custom_cmap.html
blue_red_cmap = {
    'red':  ((0.00, 0.0, 0.0),
             (0.25, 0.0, 0.0),
             (0.50, 0.8, 1.0),
             (0.75, 1.0, 1.0),
             (1.00, 0.4, 1.0)),

    'green': ((0.00, 0.0, 0.0),
              (0.25, 0.0, 0.0),
              (0.50, 0.9, 0.9),
              (0.75, 0.0, 0.0),
              (1.00, 0.0, 0.0)),

    'blue':  ((0.00, 0.0, 0.4),
              (0.25, 1.0, 1.0),
              (0.50, 1.0, 0.8),
              (0.75, 0.0, 0.0),
              (1.00, 0.0, 0.0))
}


def draw_countour(xgrid, ygrid, target_function):
    output = np.zeros((xgrid.shape[0], ygrid.shape[0]))

    for i, x in enumerate(xgrid):
        for j, y in enumerate(ygrid):
            output[j, i] = target_function(x, y)

    X, Y = np.meshgrid(xgrid, ygrid)

    plt.register_cmap(name='BlueRed', data=blue_red_cmap)
    plt.contourf(X, Y, output, 50, alpha=.75, cmap='BlueRed')
    plt.colorbar()


def weight_quiver(weights, color='c'):
    plt.quiver(weights[0, :-1],
               weights[1, :-1],
               weights[0, 1:] - weights[0, :-1],
               weights[1, 1:] - weights[1, :-1],
               scale_units='xy', angles='xy', scale=1,
               color=color)
