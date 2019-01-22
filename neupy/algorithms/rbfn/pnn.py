import numpy as np

from neupy.utils import format_data, iters
from neupy.core.properties import BoundedProperty, IntProperty
from neupy.algorithms.base import BaseSkeleton
from neupy.exceptions import NotTrained
from .utils import pdf_between_data


__all__ = ('PNN',)


class PNN(BaseSkeleton):
    """
    Probabilistic Neural Network (PNN). Network applies only to
    the classification problems.

    Notes
    -----
    - PNN Network is sensitive for cases when one input feature
      has higher values than the other one. Input data has to be
      normalized before training.

    - Standard deviation has to match the range of the input features
      Check ``std`` parameter description for more information.

    - The bigger training dataset the slower prediction.
      Algorithm is much more efficient for small datasets.

    - Network uses lazy learning which mean that network doesn't
      need iterative training. It just stores parameters
      and use them to make a predictions.

    Parameters
    ----------
    std : float
        Standard deviation for the Probability Density Function (PDF).
        If your input features have high values than standard deviation
        should also be high. For instance, if input features from range
        ``[0, 20]`` that standard deviation should be also a big value
        like ``10`` or ``15``. Small values will lead to bad prediction.

    batch_size : int or None
        Set up min-batch size. The ``None`` value will ensure that all data
        samples will be propagated through the network at once.
        Defaults to ``128``.

    {Verbose.verbose}

    Methods
    -------
    train(X_train, y_train, copy=True)
        Network just stores all the information about the data and use
        it for the prediction. Parameter ``copy`` copies input data
        before saving it inside the network.

        The ``y_train`` argument should be a vector or
        matrix with one feature column.

    predict(X)
        Return classes associated with each sample in the ``X``.

    predict_proba(X)
        Predict probabilities for each class.

    {BaseSkeleton.fit}

    Examples
    --------
    >>> import numpy as np
    >>>
    >>> from sklearn import datasets, metrics
    >>> from sklearn.model_selection import train_test_split
    >>> from neupy import algorithms
    >>>
    >>> dataset = datasets.load_digits()
    >>> x_train, x_test, y_train, y_test = train_test_split(
    ...     dataset.data, dataset.target, test_size=0.3
    ... )
    >>>
    >>> pnn = algorithms.PNN(std=10, verbose=False)
    >>> pnn.train(x_train, y_train)
    >>>
    >>> y_predicted = pnn.predict(x_test)
    >>> metrics.accuracy_score(y_test, y_predicted)
    0.98888888888888893
    """
    std = BoundedProperty(minval=0)
    batch_size = IntProperty(default=128, minval=0, allow_none=True)

    def __init__(self, std, batch_size=128, verbose=False):
        self.std = std
        self.batch_size = batch_size

        self.classes = None
        self.X_train = None
        self.y_train = None

        super(PNN, self).__init__(batch_size=batch_size, verbose=verbose)

    def train(self, X_train, y_train, copy=True):
        """
        Trains network. PNN doesn't actually train, it just stores
        input data and use it for prediction.

        Parameters
        ----------
        X_train : array-like (n_samples, n_features)

        y_train : array-like (n_samples,)
            Target variable should be vector or matrix
            with one feature column.

        copy : bool
            If value equal to ``True`` than input matrices will
            be copied. Defaults to ``True``.

        Raises
        ------
        ValueError
            In case if something is wrong with input data.
        """
        X_train = format_data(X_train, copy=copy)
        y_train = format_data(y_train, copy=copy, make_float=False)

        self.X_train = X_train
        self.y_train = y_train

        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                "Number of samples in the input and "
                "target datasets are different")

        if y_train.shape[1] != 1:
            raise ValueError(
                "Target value should be vector or "
                "matrix with only one column")

        classes = self.classes = np.unique(y_train)
        n_classes = classes.size
        n_samples = X_train.shape[0]

        class_ratios = self.class_ratios = np.zeros(n_classes)
        row_comb_matrix = self.row_comb_matrix = np.zeros(
            (n_classes, n_samples))

        for i, class_name in enumerate(classes):
            class_name = classes[i]
            class_val_positions = (y_train == class_name)
            row_comb_matrix[i, class_val_positions.ravel()] = 1
            class_ratios[i] = np.sum(class_val_positions)

    def predict_proba(self, X):
        """
        Predict probabilities for each class.

        Parameters
        ----------
        X : array-like (n_samples, n_features)

        Returns
        -------
        array-like (n_samples, n_classes)
        """
        outputs = iters.apply_batches(
            function=self.predict_raw,
            inputs=format_data(X),
            batch_size=self.batch_size,
            show_progressbar=self.logs.enable,
        )
        raw_output = np.concatenate(outputs, axis=1)

        total_output_sum = raw_output.sum(axis=0).reshape((-1, 1))
        return raw_output.T / total_output_sum

    def predict_raw(self, X):
        """
        Raw prediction.

        Parameters
        ----------
        X : array-like (n_samples, n_features)

        Raises
        ------
        NotTrained
            If network hasn't been trained.

        ValueError
            In case if something is wrong with input data.

        Returns
        -------
        array-like (n_samples, n_classes)
        """
        if self.classes is None:
            raise NotTrained(
                "Cannot make a prediction. Network hasn't been trained yet")

        if X.shape[1] != self.X_train.shape[1]:
            raise ValueError(
                "Input data must contain {0} features, got {1}"
                "".format(self.X_train.shape[1],  X.shape[1]))

        class_ratios = self.class_ratios.reshape((-1, 1))
        pdf_outputs = pdf_between_data(self.X_train, X, self.std)

        return np.dot(self.row_comb_matrix, pdf_outputs) / class_ratios

    def predict(self, X):
        """
        Predicts class from the input data.

        Parameters
        ----------
        X : array-like (n_samples, n_features)

        Returns
        -------
        array-like (n_samples,)
        """
        outputs = iters.apply_batches(
            function=self.predict_raw,
            inputs=format_data(X),
            batch_size=self.batch_size,
            show_progressbar=self.logs.enable,
        )

        raw_output = np.concatenate(outputs, axis=1)
        return self.classes[raw_output.argmax(axis=0)]
