import numpy as np

from neupy import init
from neupy.utils import format_data
from neupy.exceptions import NotTrained
from neupy.algorithms.base import BaseNetwork
from neupy.core.properties import IntProperty, Property, TypedListProperty


__all__ = ('LVQ',)


def neg_euclid_distance(input_data, weight):
    """
    Negative Euclidian distance between input
    data and weight.

    Parameters
    ----------
    input_data : array-like
        Input data.

    weight : array-like
        Neural network's weights.

    Returns
    -------
    array-like
    """
    input_data = np.expand_dims(input_data, axis=0)
    euclid_dist = np.linalg.norm(input_data - weight, axis=1)
    return -np.expand_dims(euclid_dist, axis=0)


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
        of subclasses equal to the number of classes. Defaults to ``None``
        (or the same as ``n_classes``).

    n_classes : int
        Number of classes in the data set.

    prototypes_per_class : list, None
        Defines number of prototypes per each class. For instance,
        if ``n_classes=3`` and ``n_subclasses=8`` then there are
        can be 3 subclasses for the first class, 3 for the second one
        and 2 for the third one (3 + 3 + 2 == 8). The following example
        can be specified as ``prototypes_per_class=[3, 3, 2]``.

        There are two rules that apply to this parameter:

        1. ``sum(prototypes_per_class) == n_subclasses``

        2. ``len(prototypes_per_class) == n_classes``

        The ``None`` value will distribute approximately equal
        number of subclasses per each class. It's approximately,
        because in casses when ``n_subclasses % n_classes != 0``
        there is no way to distribute equal number of subclasses
        per each class.

        Defaults to ``None``.

    {BaseNetwork.step}

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
      distance to find clusters
    - Training error is just a ratio of miscassified samples
    """
    n_inputs = IntProperty(minval=1)
    n_subclasses = IntProperty(minval=2, default=None, allow_none=True)
    n_classes = IntProperty(minval=2)

    prototypes_per_class = TypedListProperty(allow_none=True, default=None)
    weight = Property(expected_type=(np.ndarray, init.Initializer),
                      allow_none=True, default=None)

    def __init__(self, **options):
        self.initialized = False
        super(LVQ, self).__init__(**options)

        if self.n_subclasses is None:
            self.n_subclasses = self.n_classes

        if isinstance(self.weight, init.Initializer):
            weight_shape = (self.n_inputs, self.n_subclasses)
            self.weight = self.weight.sample(weight_shape)

        if self.weight is not None:
            self.initialized = True

        if self.n_subclasses < self.n_classes:
            raise ValueError("Number of subclasses should be greater "
                             "or equal to the number of classes. Network "
                             "was defined with {} subclasses and {} classes"
                             "".format(self.n_subclasses, self.n_classes))

        if self.prototypes_per_class is None:
            whole, reminder = divmod(self.n_subclasses, self.n_classes)
            self.prototypes_per_class = [whole] * self.n_classes

            if reminder:
                # Since we have reminder left, it means that we cannot
                # have an equal number of subclasses per each class,
                # therefor we will add +1 to randomly selected class.
                class_indeces = np.random.choice(self.n_classes, reminder,
                                                 replace=False)

                for class_index in class_indeces:
                    self.prototypes_per_class[class_index] += 1

        if len(self.prototypes_per_class) != self.n_classes:
            raise ValueError("LVQ defined for classification problem that has "
                             "{} classes, but the `prototypes_per_class` "
                             "variable has defined data for {} classes."
                             "".format(self.n_classes,
                                       len(self.prototypes_per_class)))

        if sum(self.prototypes_per_class) != self.n_subclasses:
            raise ValueError("Invalid distribution of subclasses for the "
                             "`prototypes_per_class` variable. Got total "
                             "of {} subclasses ({}) instead of {} expected"
                             "".format(sum(self.prototypes_per_class),
                                       self.prototypes_per_class,
                                       self.n_subclasses))

        self.subclass_to_class = []
        for class_id, n_prototypes in enumerate(self.prototypes_per_class):
            self.subclass_to_class.extend([class_id] * n_prototypes)

    def predict(self, input_data):
        if not self.initialized:
            raise NotTrained("LVQ network hasn't been trained yet")

        input_data = format_data(input_data)
        subclass_to_class = self.subclass_to_class
        weight = self.weight

        predictions = []
        for input_row in input_data:
            output = neg_euclid_distance(input_row, weight)
            winner_subclass = int(output.argmax(axis=1))

            predicted_class = subclass_to_class[winner_subclass]
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
            target_classes = sorted(np.unique(target_train).astype(np.int))
            expected_classes = list(range(self.n_classes))

            if target_classes != expected_classes:
                raise ValueError("All classes should be integers from the "
                                 "range [0, {}], but got the following "
                                 "classes instead {}".format(
                                    self.n_classes - 1, target_classes))

            weights = []
            iterator = zip(target_classes, self.prototypes_per_class)
            for target_class, n_prototypes in iterator:
                is_valid_class = (target_train[:, 0] == target_class)
                is_valid_class = is_valid_class.astype('float64')
                n_samples_per_class = sum(is_valid_class)
                is_valid_class /= n_samples_per_class

                if n_samples_per_class <= n_prototypes:
                    raise ValueError("Input data has {0} samples for class-{1}"
                                     ". Number of samples per specified "
                                     "class-{1} should be greater than {2}."
                                     "".format(n_samples_per_class,
                                               target_class, n_prototypes))

                class_weight_indeces = np.random.choice(
                    np.arange(n_input_samples), n_prototypes,
                    replace=False, p=is_valid_class)

                class_weight = input_train[class_weight_indeces]
                weights.extend(class_weight)

            self.weight = np.array(weights)
            self.initialized = True

        super(LVQ, self).train(input_train, target_train, *args, **kwargs)

    def train_epoch(self, input_train, target_train):
        step = self.step
        weight = self.weight
        subclass_to_class = self.subclass_to_class

        n_correct_predictions = 0
        for input_row, target in zip(input_train, target_train):
            output = neg_euclid_distance(input_row, weight)
            winner_subclass = int(output.argmax())
            predicted_class = subclass_to_class[winner_subclass]

            weight_update = input_row - weight[winner_subclass, :]
            is_correct_prediction = (predicted_class == target)

            if is_correct_prediction:
                weight[winner_subclass, :] += step * weight_update
            else:
                weight[winner_subclass, :] -= step * weight_update

            n_correct_predictions += is_correct_prediction

        n_samples = len(input_train)
        return 1 - n_correct_predictions / n_samples
