from neupy.core.docs import SharedDocs
from neupy.utils import format_data


__all__ = ('SupervisedLearningMixin', 'LazyLearningMixin')


class SupervisedLearningMixin(object):
    """
    Mixin for Supervised Neural Network algorithms.

    Methods
    -------
    train(input_train, target_train, input_test=None, target_test=None,\
    epochs=100, epsilon=None)
        Train network. You can control network's training procedure
        with ``epochs`` and ``epsilon`` parameters.
        The ``input_test`` and ``target_test`` should be presented
        both in case of you need to validate network's training
        after each iteration.
    """
    def train(self, input_train, target_train, input_test=None,
              target_test=None, epochs=100, epsilon=None,
              summary='table'):

        is_test_data_partialy_missed = (
            (input_test is None and target_test is not None) or
            (input_test is not None and target_test is None)
        )

        if is_test_data_partialy_missed:
            raise ValueError("Input and target test samples are missed. "
                             "They must be defined together or none of them.")

        input_train = format_data(input_train)
        target_train = format_data(target_train)

        if input_test is not None:
            input_test = format_data(input_test)

        if target_test is not None:
            target_test = format_data(target_test)

        return super(SupervisedLearningMixin, self).train(
            input_train=input_train, target_train=target_train,
            input_test=input_test, target_test=target_test,
            epochs=epochs, epsilon=epsilon,
            summary=summary
        )


class LazyLearningMixin(SharedDocs):
    """
    Mixin for lazy learning Neural Network algorithms.

    Notes
    -----
    * Network uses lazy learning which mean that network doesn't \
    involve iterative training procedure. It just stores parameters \
    and use them to make a predictions.

    Methods
    -------
    train(input_train, target_train, copy=True)
        Network just stores all the information about the data and use \
        it for the prediction. Parameter ``copy`` copy input data \
        before store it inside the network.
    """
    def __init__(self, *args, **kwargs):
        self.input_train = None
        self.target_train = None
        super(LazyLearningMixin, self).__init__(*args, **kwargs)

    def init_properties(self):
        del self.shuffle_data
        del self.step
        del self.show_epoch
        del self.train_end_signal
        del self.epoch_end_signal

    def train(self, input_train, target_train):
        self.input_train = input_train
        self.target_train = target_train

        if input_train.shape[0] != target_train.shape[0]:
            raise ValueError("Number of samples in the input and target "
                             "datasets are different")

    def train_epoch(self, *args, **kwargs):
        raise AttributeError("This network doesn't have train epochs")

    def predict(self, input_data):
        if self.input_train is None:
            raise ValueError("Network hasn't been trained yet")
