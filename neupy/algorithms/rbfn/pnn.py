import numpy as np

from neupy.utils import format_data
from neupy.exceptions import NotTrained
from neupy.core.properties import BoundedProperty
from neupy.algorithms.base import BaseSkeleton
from neupy.algorithms.gd.base import MinibatchTrainingMixin
from .utils import pdf_between_data


__all__ = ('PNN',)


class PNN(BaseSkeleton, MinibatchTrainingMixin):
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
        Defaults to ``0.1``. If your input features have high values
        than standard deviation should also be high. For instance,
        if input features from range ``[0, 20]`` that standard
        deviation should be also a big value like ``10`` or ``15``.
        Small values will lead to bad prediction.

    {MinibatchTrainingMixin.batch_size}

    {Verbose.verbose}

    Methods
    -------
    train(X_train, y_train, copy=True)
        Network just stores all the information about the data and use
        it for the prediction. Parameter ``copy`` copies input data
        before saving it inside the network.

        The ``y_train`` argument should be a vector or
        matrix with one feature column.

    {BaseSkeleton.predict}

    predict_proba(X)
        Predict probabilities for each class.

    {BaseSkeleton.fit}

    Examples
    --------
    >>> import numpy as np
    >>>
    >>> from sklearn import datasets, metrics
    >>> from sklearn.model_selection import train_test_split
    >>> from neupy import algorithms, environment
    >>>
    >>> environment.reproducible()
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
    std = BoundedProperty(default=0.1, minval=0)

    def __init__(self, **options):
        self.classes = None
        self.X_train = None
        self.y_train = None

        super(PNN, self).__init__(**options)

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
            raise ValueError("Number of samples in the input and target "
                             "datasets are different")

        n_target_features = y_train.shape[1]
        if n_target_features != 1:
            raise ValueError("Target value should be a vector or a "
                             "matrix with one column")

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
        outputs = self.apply_batches(
            function=self.predict_raw,
            X=format_data(X),

            description='Prediction batches',
            show_progressbar=True,
            show_error_output=False,
            scalar_output=False,
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
            raise NotTrained("Cannot make a prediction. Network "
                             "hasn't been trained yet")

        X_size = X.shape[1]
        train_data_size = self.X_train.shape[1]

        if X_size != train_data_size:
            raise ValueError("Input data must contain {0} features, got "
                             "{1}".format(train_data_size, X_size))

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
        outputs = self.apply_batches(
            function=self.predict_raw,
            X=format_data(X),

            description='Prediction batches',
            show_progressbar=True,
            show_error_output=False,
            scalar_output=False,
        )

        raw_output = np.concatenate(outputs, axis=1)
        return self.classes[raw_output.argmax(axis=0)]
