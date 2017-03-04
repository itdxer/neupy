import numpy as np

from neupy import init
from neupy.utils import format_data
from neupy.algorithms.base import BaseNetwork
from neupy.core.properties import IntProperty, NumberProperty, Property
from .sofm import neg_euclid_distance


__all__ = ('LVQ',)


class LVQ(BaseNetwork):
    """
    Learning Vector Quantization (LVQ) algorithm.

    Parameters
    ----------
    n_inputs : int
        Number of input units. It should be equal to the
        number of features in the input data set.

    n_subclasses : int, None
        Defines total number of subclasses. Values should be greater
        or equal to the number of classes. ``None`` will set up number
        of subclasses equal to the number of classes.

    n_classes : int
        Number of classes in the data set.

    prototypes_per_class : list or {{'take-equal', 'take-proportional'}}
        Defines number of prototypes per each class. For instance,
        if ``n_classes=3`` and ``n_subclasses=8`` then there are
        can be 3 subclasses for the first class, 3 for the second one
        and 2 for the third one (3 + 3 + 2 == 8). The following example
        can be specified as ``prototypes_per_class=[3, 3, 2]``.

    {BaseNetwork.step}

    step_decrease_rate : float
        Value defines decrease rate of the step after each epoch.
        Step decreasing procedure would be applied until ``step`` would
        be equal to value specified  for the ``minstep`` parameter.

    minstep : float
        Defaults to ``1e-5``.

    {BaseNetwork.show_epoch}

    {BaseNetwork.shuffle_data}

    {BaseNetwork.epoch_end_signal}

    {BaseNetwork.train_end_signal}

    {Verbose.verbose}

    Methods
    -------
    {BaseSkeleton.predict}

    {BaseSkeleton.fit}

    Notes
    -----
    - Input data needs to be normalized, because LVQ uses Euclidian
      distance to find
    """
    n_inputs = IntProperty(minval=1)
    n_subclasses = IntProperty(minval=2, default=None)
    n_classes = IntProperty(minval=2)

    weight = Property(expected_type=(np.ndarray, init.Initializer),
                      allow_none=True, default=None)

    minstep = NumberProperty(default=1e-5)

    def __init__(self, **options):
        self.initialized = False
        super(LVQ, self).__init__(**options)

        if self.weight is not None:
            self.initialized = True

        if self.n_subclasses < self.n_classes:
            raise ValueError("Number of subclasses should be greater "
                             "or equal to the number of classes. Network "
                             "defined with {} subclasses and {} classes"
                             "".format(self.n_subclasses, self.n_classes))

        # TODO: init and validate `prototypes_per_class`
        # if sum(self.prototypes_per_class) != self.n_subclasses:
        #     raise ValueError("...")

    def predict(self, input_data):
        if not self.initialized:
            raise NotTrained("LVQ network hasn't been trained yet")

        input_data = format_data(input_data)
        weight = self.weight

        predictions = []
        for input_row in input_data:
            output = neg_euclid_distance(input_row, weight)
            winner_subclass = output.argmax(axis=1)

            predicted_class = subclass_mapping[winner_subclass]
            predictions.append(predicted_class)

        return np.array(predictions)

    def train(self, input_train, target_train, *args, **kwargs):
        input_train = format_data(input_train)
        target_train = format_data(target_train)

        n_input_samples = len(input_train)

        if n_input_samples <= self.n_subclasses:
            raise ValueError("Number of training input samples should be "
                             "greater than number of sublcasses. Training "
                             "method recived {} input samples."
                             "".format(n_input_samples))

        if not self.initialized:
            target_classes = sorted(np.unique(target_train, dtype='int'))
            expected_classes = list(range(self.n_classes))

            if target_classes != expected_classes:
                raise ValueError("...")

            weights = []
            iterator = zip(target_classes, self.prototypes_per_class)
            for target_class, n_prototypes in iterator:
                is_valid_class = (target_train == target_class)
                n_samples_per_class = sum(is_valid_class)

                if n_samples_per_class <= n_prototypes:
                    raise ValueError("Input data has {0} samples for class "
                                     "{1}. Number of samples per specified "
                                     "class {1} should be greater than {2}."
                                     "".format(n_samples_per_class,
                                               target_class, n_prototypes))

                class_weight = np.random.choice(
                    input_train, n_prototypes, replace=False,
                    p=is_valid_class / n_samples_per_class)

                weights.extend(weights)

            self.weights = np.array(weights)
            self.initialized = True

        super(LVQ, self).train(input_train, target_train, *args, **kwargs)

    def train_epoch(self, input_train, target_train):
        step = self.step
        weight = self.weight
        predict = self.predict

        for input_row, target in zip(input_train, target_train):
            input_row = np.reshape(input_row, (1, input_row.size))

            output = neg_euclid_distance(input_row, weight)
            winner_subclass = output.argmax(axis=1)
            predicted_class = subclass_mapping[winner_subclass]

            if predicted_class == target:
                weight[:, winner_subclass] += step * (
                    input_row.T - weight[:, winner_subclass])
            else:
                weight[:, winner_subclass] -= step * (
                    input_row.T - weight[:, winner_subclass])

        # TODO: decrease training step
