from __future__ import division

import numpy as np

from neupy.core.docs import shared_docs


def gaussian_df(data, mean=0, std=1):
    """
    Returns gaussian density for each data sample.
    Gaussian specified by the mean and standard deviation.

    Parameters
    ----------
    data : array-like

    mean : float
        Gaussian mean.

    std : float
        Gaussian standard deviation.
    """
    if std == 0:
        return np.where(data == mean, 1, 0)

    normalizer = 2 * np.pi * std ** 2
    return np.exp(-np.square(data - mean) / normalizer)


def find_neighbour_distance(grid, center):
    """
    Returns distance from the center into different directions
    per each dimension separately.

    Parameters
    ----------
    grid : array-like
       Array that contains grid of n-dimensional vectors.

    center : tuple
        Index of the main neuron for which function returns
        distance to neuron's neighbours.

    Returns
    -------
    list of n-dimensional vectors
    """
    if len(center) != grid.ndim:
        raise ValueError(
            "Cannot find center, because grid of neurons has {} dimensions "
            "and center has specified coordinates for {} dimensional grid"
            "".format(grid.ndim, len(center)))

    slices = []
    for dim_length, center_coord in zip(grid.shape, center):
        slices.append(slice(-center_coord, dim_length - center_coord))

    return np.ogrid[slices]


@shared_docs(find_neighbour_distance)
def find_step_scaler_on_rect_grid(grid, center, std=1):
    """
    Function returns multivariate gaussian around the center
    with specified standard deviation.

    Parameters
    ----------
    {find_neighbour_distance.grid}

    {find_neighbour_distance.center}

    std : int, float
        Gaussian standard deviation. Defaults to ``1``.
    """
    distances = find_neighbour_distance(grid, center)
    gaussian_array = sum(gaussian_df(dist, std=std) for dist in distances)
    return gaussian_array / grid.ndim


@shared_docs(find_neighbour_distance)
def find_neighbours_on_rect_grid(grid, center, radius):
    """
    Function find all neuron's neighbours around specified
    center within a certain radius.

    Parameters
    ----------
    {find_neighbour_distance.grid}

    {find_neighbour_distance.center}

    radius : int
        Radius specifies what neurons, around the center, are neighbours.

    Returns
    -------
    array-like
        Return matrix with the same dimension as ``grid``
        where center element and it neighbours positions
        filled with value ``1`` and other as a ``0`` value.

    Examples
    --------
    >>> import numpy as np
    >>> from neupy.algorithms.competitive import sofm
    >>>
    >>> sofm.find_neighbours_on_rect_grid(
    ...     grid=np.zeros((3, 3)),
    ...     center=(0, 0),
    ...     radius=1)
    ...
    array([[ 1.,  1.,  0.],
           [ 1.,  0.,  0.],
           [ 0.,  0.,  0.]])
    >>>
    >>> sofm.find_neighbours_on_rect_grid(
    ...     grid=np.zeros((5, 5)),
    ...     center=(2, 2),
    ...     radius=2)
    ...
    array([[ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  0.,  1.,  0.,  0.]])
    """
    distances = find_neighbour_distance(grid, center)
    mask = sum(dist ** 2 for dist in distances) <= radius ** 2
    grid[mask] = 1
    return grid


def generate_neighbours_pattern(radius):
    """
    Generates pattern that defines neighbours
    within specified radius on the hexagon grid.

    Parameters
    ----------
    radius : int

    Returns
    -------
    array-like
        2d-array that has number of rows equal to number
        of columns equal to ``2 * radius + 1``
    """
    cache = generate_neighbours_pattern.cache

    if radius in cache:
        return cache[radius]

    size = 2 * radius + 1
    max_distance = radius + 1
    pattern = np.zeros((size, size))

    for i, width in enumerate(range(size - radius, size + 1)):
        start_point = radius - int(np.floor(width / 2))
        neighbour_range = slice(start_point, start_point + width)

        neighbour_distance = np.pad(
            array=(max_distance - i) * [max_distance - i],
            pad_width=i,
            mode='linear_ramp',
            end_values=max_distance)

        pattern[i, neighbour_range] = neighbour_distance
        # every row in the pattern is mirror symmetric
        pattern[size - i - 1, neighbour_range] = neighbour_distance

    cache[radius] = pattern
    return pattern


generate_neighbours_pattern.cache = {}


def find_neighbours_on_hexagon_grid(grid, center, radius):
    """
    Marks neighbors on the hexagon grid around specified
    center within specified radius.

    Parameters
    ----------
    grid : 2d-arry
        Hexagon grid.

    center : tuple
        Coordinates of the center neuron on the grid.
        Should be a tuple with two integers: ``(x, y)``.

    radius : int
        Radius specifies what neurons, around the center, are neighbours.

    Returns
    -------
    2d-array
    """
    if radius == 0:
        grid[center] = 1
        return grid

    center_x, center_y = center
    pattern = generate_neighbours_pattern(radius)

    # We add zero padding in order to be able
    # insert full pattern into the matrix. It's
    # useful in case if we are not able to mark all
    # neighbours from the pattern for specific neuron.
    # For instance, in case when center = (0, 0) we
    # are not able to mark neurons on the left and top
    # sides of the pattern's center.
    grid = np.pad(grid, radius, mode='constant')

    if center_x % 2 == 1:
        # Since every even row shifted to the right
        # compare to the odd rows then odd rows shifted
        # to the left compare to the even rows.
        # It means that our pattern has to be reversed
        # for the even rows.
        pattern = pattern[:, ::-1]

    pattern_size = 2 * radius + 1
    x_range = slice(center_x, center_x + pattern_size)
    y_range = slice(center_y, center_y + pattern_size)

    grid[x_range, y_range] = pattern

    # removing padding
    return grid[radius:-radius, radius:-radius]


@shared_docs(find_step_scaler_on_rect_grid)
def find_step_scaler_on_hexagon_grid(grid, center=None, std=1):
    """
    Find step scale for neighbouring neurons. The further
    neuron from the center the lower step scale it has.

    Notes
    -----
    Non-neighbor neurons will have high step size
    We do not set up it to zero, since later we will
    ignore these values anyway

    Parameters
    ----------
    grid : array-like

    center : object
        Variable is going to be ignored. It's defined
        only for compatibility with other functions

    {find_step_scaler_on_rect_grid.std}
    """
    return gaussian_df(grid, mean=1, std=std)
