import pkgutil
import warnings
import importlib
import numpy

from PIL import Image

import matplotlib.pyplot as plt


def load_pandas_module():
    if not pkgutil.find_loader('pandas'):
        raise ImportError(
            "The `pandas` library is not installed. Try to "
            "install it with pip: \n    pip install pandas")

    return importlib.import_module('pandas')


def plot(x_train, y_train, x_valid, y_valid, ax, logx=False):
    plot_function = ax.semilogx if logx else ax.plot

    # With large number of samples there will be to many
    # points in the plot and it will be hard to read it
    style = '-' if len(x_train) > 50 or len(x_valid) > 50 else 'o-'
    line_train, = plot_function(x_train, y_train, style, label='train')

    if len(x_valid) > 0:
        line_valid_, = plot_function(x_valid, y_valid, style, label='valid')
        ax.legend([line_train, line_valid_], ['Training', 'Validation'])


def plot_errors_per_update(train, valid, ax, logx=False):
    plot(
        # Loss that we've obtain per each training update is calculated before
        # the update. Which means that loss represents state of the network
        # before the update.
        x_train=train.n_updates - 1,
        y_train=train.value,

        x_valid=valid.n_updates,
        y_valid=valid.value,

        logx=logx,
        ax=ax,
    )

    ax.set_title('Training perfomance')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Number of updates passed')


def plot_error_per_epoch(train, valid, ax, logx=False):
    valid = valid[['epoch', 'value']]
    train = train[['epoch', 'value']]

    train_epochs = train.groupby('epoch').agg({'value': 'first'})
    valid_epochs = valid.groupby('epoch').agg({'value': 'first'})

    plot(
        # Loss that we've obtain per each training update is calculated before
        # the update. Which means that loss represents state of the network
        # before the update.
        x_train=train_epochs.index - 1,
        y_train=train_epochs.value,

        x_valid=valid_epochs.index,
        y_valid=valid_epochs.value,

        logx=logx,
        ax=ax,
    )

    ax.set_ylabel('Loss')
    ax.set_xlabel('Number of training epochs passed')


def plot_optimizer_errors(optimizer, logx=False, show=True, image=False, **figkwargs):
    if image:
        show=False
    if 'figsize' not in figkwargs:
        figkwargs['figsize'] = (12, 8)

    if not optimizer.events.logs:
        warnings.warn("There is no data to plot")
        return

    pd = load_pandas_module()
    history = pd.DataFrame(optimizer.events.logs)

    train = history[history.name == 'train_error']
    valid = history[history.name == 'valid_error']
    train_max = train.max()

    if train_max.epoch == train_max.n_updates:
        fig, ax = plt.subplots(1, 1, **figkwargs)
        # When number of epochs exactly the same as number of updates
        # plots per number of updates and epochs will be exactly the same
        plot_error_per_epoch(train, valid, logx=logx, ax=ax)
    else:
        fig, axes = plt.subplots(2, 1, **figkwargs)
        plot_errors_per_update(train, valid, logx=logx, ax=axes[0])
        plot_error_per_epoch(train, valid, logx=logx, ax=axes[1])

    if show:
        plt.show()

    if image:
        return figure_to_image(fig)



def figure_to_data(figure):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param figure - a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    figure.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = figure.canvas.get_width_height()
    np_array = numpy.fromstring(figure.canvas.tostring_argb(), dtype=numpy.uint8)
    np_array.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    np_array = numpy.roll(np_array, 3, axis=2)
    return np_array


def figure_to_image(figure):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = figure_to_data(figure)
    w, h, d = buf.shape
    return Image.frombytes("RGBA", (w, h), buf.tostring())
