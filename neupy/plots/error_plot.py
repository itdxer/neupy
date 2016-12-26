import numpy as np
import matplotlib.pyplot as plt


__all__ = ('error_plot',)


def error_plot(network, logx=False, ax=None, show=True):
    """
    Makes line plot that shows training progress. x-axis
    is an epoch number and y-axis is an error.

    Parameters
    ----------
    logx : bool
        Parameter set up logarithmic scale to x-axis.
        Defaults to ``False``.

    ax : object or None
        Matplotlib axis object. ``None`` values means that axis equal
        to the current one (the same as ``ax = plt.gca()``).
        Defaults to ``None``.

    show : bool
        If parameter is equal to ``True`` then plot will be
        displayed. Defaults to ``True``.

    Returns
    -------
    object
        Matplotlib axis instance.

    Examples
    --------
    >>> from neupy import algorithms, plots
    >>>
    >>> gdnet = algorithms.GradientDescent((2, 3, 1))
    >>> gdnet.train(x_train, y_train, x_test, y_test, epochs=100)
    >>>
    >>> plots.error_plot(gdnet)
    """
    if ax is None:
        ax = plt.gca()

    if not network.errors:
        network.logs.warning("There is no data to plot")
        return ax

    train_errors = network.errors.normalized()
    validation_errors = network.validation_errors.normalized()

    if len(train_errors) != len(validation_errors):
        network.logs.warning("Mismatch in number of training and validation "
                             "errors. Validation error will be ignored.")
        validation_errors = []

    if all(err is None for err in validation_errors):
        validation_errors = []

    errors_range = np.arange(len(train_errors))
    plot_function = ax.semilogx if logx else ax.plot

    line_error_in, = plot_function(errors_range, train_errors)

    if validation_errors:
        line_error_out, = plot_function(errors_range, validation_errors)
        ax.legend(
            [line_error_in, line_error_out],
            ['Train', 'Validation']
        )

    ax.set_title('Training perfomance')
    ax.set_ylim(bottom=0)

    ax.set_ylabel('Error')
    ax.set_xlabel('Epoch')

    if show:
        plt.show()

    return ax
