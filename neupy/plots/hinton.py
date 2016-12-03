import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from neupy.utils import format_data


__all__ = ('hinton',)


def hinton(matrix, max_weight=None, ax=None, add_legend=True):
    """
    Draw Hinton diagram for visualizing a weight matrix.

    Parameters
    ----------
    matrix: array-like
        Matrix that you want to visualise using Hinton diagram.

    max_weight : float
        Maximum value of the matrix. If it's equal to ``None``
        than value would be calculated using the maximum from
        the matrix. Defaults to ``None``.

    ax : object
        Matplotlib Axes instantce. If value equal to ``None``
        then function generate the new Axes instance. Defaults
        to ``None``.

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

    References
    ----------
    [1] http://matplotlib.org/examples/specialty_plots/hinton_demo.html
    """
    if ax is None:
        ax = plt.gca()

    matrix = format_data(matrix, is_feature1d=True)

    if max_weight is None:
        max_value = np.abs(matrix).max()
        max_value_log2_base = np.log(max_value) / np.log(2)
        max_weight = 2 ** np.ceil(max_value_log2_base)

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (y, x), weight in np.ndenumerate(matrix):
        color = ('white' if weight > 0 else 'black')
        size = min(np.sqrt(np.abs(weight / max_weight)), 1.)
        rect = plt.Rectangle([x - size / 2., y - size / 2.], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()

    if add_legend:
        max_value = matrix.max().round(2)
        min_value = matrix.min().round(2)

        white = Rectangle(xy=(0, 0), width=1., height=1., linewidth=1.,
                          linestyle='solid', facecolor='#ffffff')
        black = Rectangle(xy=(0, 0), width=1., height=1., color='#000000')

        if min_value < 0 and max_value > 0:
            rectangles = [white, black]
            rect_description = [
                'Positive value\n'
                'Max: {}'.format(max_value),
                'Negative value\n'
                'Min: {}'.format(min_value),
            ]

        elif min_value >= 0:
            rectangles = [white]
            rect_description = [
                'Positive value\n'
                'Min: {}\n'
                'Max: {}'.format(min_value, max_value),
            ]

        else:
            rectangles = [black]
            rect_description = [
                'Negative value\n'
                'Min: {}\n'
                'Max: {}'.format(min_value, max_value),
            ]

        ax.legend(rectangles, rect_description, loc='center left',
                  bbox_to_anchor=(1., 0.5))

    return ax
