from numpy import dot

from neupy.utils import format_data
from neupy.core.properties import BoundedProperty
from neupy.network.base import BaseNetwork
from neupy.network.learning import LazyLearningMixin
from .utils import pdf_between_data


__all__ = ('GRNN',)


class GRNN(BaseNetwork, LazyLearningMixin):
    """
    Generalized Regression Neural Network (GRNN). Network applies
    only to the regression problems.

    Parameters
    ----------
    std : float
        Standard deviation for PDF function, defaults to ``0.1``.
        If your input features have high values than standard
        deviation should also be high. For instance, if input features
        from range ``[0, 20]`` that standard deviation should be
        also a big value like ``10`` or ``15``. Small values will
        lead to bad prediction.
    {Verbose.verbose}

    Notes
    -----
    * GRNN Network is sensitive for cases when one input feature has \
    higher values than the other one. Before use it make sure that \
    input values are normalized and have similar scales.

    * Make sure that standard deviation in the same range as \
    input features. Check ``std`` parameter description for \
    more information.

    * The bigger training dataset the slower prediction. \
    It's much more efficient for small datasets.

    {LazyLearningMixin.Notes}

    Methods
    -------
    {LazyLearningMixin.train}
    {BaseSkeleton.predict}
    {BaseSkeleton.fit}

    Examples
    --------
    >>> from sklearn import datasets, preprocessing
    >>> from sklearn.cross_validation import train_test_split
    >>> from neupy import algorithms, estimators, environment
    >>>
    >>> environment.reproducible()
    >>>
    >>> dataset = datasets.load_diabetes()
    >>> x_train, x_test, y_train, y_test = train_test_split(
    ...     preprocessing.minmax_scale(dataset.data),
    ...     preprocessing.minmax_scale(dataset.target.reshape((-1, 1))),
    ...     train_size=0.7,
    ... )
    >>>
    >>> nw = algorithms.GRNN(std=0.1, verbose=False)
    >>> nw.train(x_train, y_train)
    >>> result = nw.predict(x_test)
    >>> estimators.rmse(result, y_test)
    0.2381013391408185
    """
    std = BoundedProperty(default=0.1, minval=0)

    def train(self, input_train, target_train, copy=True):
        """
        Trains network. PNN doesn't actually train, it just stores
        input data and use it for prediction.

        Parameters
        ----------
        input_train : array-like (n_samples, n_features)
        target_train : array-like (n_samples,)
        copy : bool
            If value equal to ``True`` than input matrices will
            be copied. Defaults to ``True``.

        Raises
        ------
        ValueError
            In case if something is wrong with input data.
        """
        input_train = format_data(input_train, copy=copy)
        target_train = format_data(target_train, copy=copy)

        if target_train.shape[1] != 1:
            raise ValueError("Target value must be one dimensional array")

        LazyLearningMixin.train(self, input_train, target_train)

    def predict(self, input_data):
        """
        Make a prediction from the input data.

        Parameters
        ----------
        input_data : array-like (n_samples, n_features)

        Raises
        ------
        ValueError
            In case if something is wrong with input data.

        Returns
        -------
        array-like (n_samples,)
        """
        super(GRNN, self).predict(input_data)

        input_data = format_data(input_data)

        input_data_size = input_data.shape[1]
        train_data_size = self.input_train.shape[1]

        if input_data_size != train_data_size:
            raise ValueError("Input data must contains {0} features, got "
                             "{1}".format(train_data_size, input_data_size))

        ratios = pdf_between_data(self.input_train, input_data, self.std)
        return (dot(self.target_train.T, ratios) / ratios.sum(axis=0)).T
