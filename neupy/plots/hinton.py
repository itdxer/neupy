"""
Code source:
http://matplotlib.org/examples/specialty_plots/hinton_demo.html
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


__all__ = ('hinton',)


def hinton(matrix, max_weight=None, ax=None, add_legend=True):
    """ Draw Hinton diagram for visualizing a weight matrix.

    Parameters
    ----------
    matrix: array like
        Matrix that you want to visualise using Hinton diagram.
    max_weight : float
        Maximum value of the matrix. If it's equal to ``None`` than value
        would be calculated using the maximum from the matrix. Defaults
        to ``None``.
    ax : object
        Matplotlib Axes instantce. If value equal to ``None`` then function
        generate the new Axes instance. Defaults to ``None``.

    Returns
    -------
    object
        Matplotlib Axes instance.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from neupy import plots
    >>>
    >>> weight = np.random.randn(20, 20)
    >>>
    >>> plt.style.use('ggplot')
    >>> plt.title("Hinton diagram")
    >>> plt.figure(figsize=(16, 12))
    >>> plots.hinton(weight)
    >>> plt.show()
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
        size = min(np.sqrt(np.abs(weight / max_weight)), 1)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()

    if add_legend:
        # Define a legend for the plot
        white = Rectangle((0, 0), 1, 1, linewidth=1, linestyle='solid',
                          facecolor='#ffffff')
        black = Rectangle((0, 0), 1, 1, color='#000000')
        ax.legend(
            [white, black],
            [
                'Positive value\nMax: {}'.format(matrix.max().round(2)),
                'Negative value\nMin: {}'.format(matrix.min().round(2))
            ],
            loc='center left',
            bbox_to_anchor=(1, 0.5)
        )

    return ax
