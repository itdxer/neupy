from neupy.utils import format_data


__all__ = ('SupervisedLearning', 'UnsupervisedLearning', 'LazyLearning')


class SupervisedLearning(object):
    """ Mixin for Supervised Neural Network algorithms.

    Methods
    -------
    train(input_train, target_train, input_test=None, target_test=None,\
    epochs=100, epsilon=None)
        Trains network. You can control network training procedure
        iterations with the number of epochs or converge value epsilon.
        Also you can specify ``input_test`` and ``target_test`` and control
        your validation data error on each iteration.
    """
    def train(self, input_train, target_train, input_test=None,
              target_test=None, epochs=100, epsilon=None,
              summary_type='table'):

        is_test_data_partialy_missed = (
            (input_test is None and target_test is not None) or
            (input_test is not None and target_test is None)
        )

        if is_test_data_partialy_missed:
            raise ValueError("Input and target test samples missed. "
                             "They must be defined both or none of them.")

        input_train = format_data(input_train)
        target_train = format_data(target_train)

        if input_test is not None:
            input_test = format_data(input_test)

        if target_test is not None:
            target_test = format_data(target_test)

        return super(SupervisedLearning, self).train(
            input_train=input_train, target_train=target_train,
            input_test=input_test, target_test=target_test,
            epochs=epochs, epsilon=epsilon,
            summary_type=summary_type
        )


class UnsupervisedLearning(object):
    """ Mixin for Unsupervised Neural Network algorithms.

    Methods
    -------
    train(input_train, epsilon=1e-5, epochs=100)
        Trains network until it converge. Parameter ``epochs`` control
        maximum number of iterations, just to make sure that network will
        stop training procedure if it can't converge.
    """
    def train(self, input_train, epochs=100, epsilon=None,
              summary_type='table'):

        input_train = format_data(input_train, is_feature1d=True)
        return super(UnsupervisedLearning, self).train(
            input_train=input_train, target_train=None,
            input_test=None, target_test=None,
            epochs=epochs, epsilon=epsilon,
            summary_type=summary_type
        )


class LazyLearning(object):
    """ Mixin for lazy learning Neural Network algorithms.

    Methods
    -------
    train(input_train, target_train, copy=True)
        Network just stores all the information about the data and use it for \
        the prediction. Parameter ``copy`` copy input data before store it \
        inside the network.
    """
    def __init__(self, *args, **kwargs):
        self.input_train = None
        self.target_train = None
        super(LazyLearning, self).__init__(*args, **kwargs)

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
            raise ValueError("Input data size should be the same as "
                             "target data size")

    def train_epoch(self, *args, **kwargs):
        raise AttributeError("This network doesn't have train epochs")

    def predict(self, input_data):
        if self.input_train is None:
            raise ValueError("Train neural network before make prediction")
