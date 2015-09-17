"""
Get code from:
http://matplotlib.org/examples/specialty_plots/hinton_demo.html
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


__all__ = ('hinton',)


def hinton(matrix, max_weight=None, ax=None):
    """ Draw Hinton diagram for visualizing a weight matrix.
    """
    if ax is None:
        ax = plt.gca()

    if max_weight is None:
        max_value = np.abs(matrix).max()
        max_value_log2_base = np.log(max_value) / np.log(2)
        max_weight = 2 ** np.ceil(max_value_log2_base)

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), weight in np.ndenumerate(matrix):
        color = 'white' if weight > 0 else 'black'
        size = np.sqrt(np.abs(weight / max_weight))
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()

    # Define a legend for the plot
    white = Rectangle((0, 0), 1, 1, linewidth=1, linestyle='solid',
                      facecolor='#ffffff')
    black = Rectangle((0, 0), 1, 1, color='#000000')
    plt.legend(
        [white, black],
        [
            'Positive value\nMax: {}'.format(matrix.max()),
            'Negative value\nMin: {}'.format(matrix.min())
        ],
        loc='center left',
        bbox_to_anchor=(1, 0.5)
    )
