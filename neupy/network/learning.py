__all__ = ('SupervisedLearning', 'UnsupervisedLearning', 'LazyLearning')


class SupervisedLearning(object):
    """ Mixin for Supervised Neural Network algorithms.
    """
    def train(self, input_train, target_train, input_test=None,
              target_test=None, epochs=None, epsilon=None):
        self._train(input_train=input_train, target_train=target_train,
                    input_test=input_test, target_test=target_test,
                    epochs=epochs, epsilon=epsilon)


class UnsupervisedLearning(object):
    """ Mixin for Unsupervised Neural Network algorithms.
    """
    def train(self, input_train, epochs=None, epsilon=None):
        self._train(input_train=input_train, target_train=None,
                    input_test=None, target_test=None,
                    epochs=epochs, epsilon=epsilon)

    def predict_prob(self, input_data):
        raise AttributeError("Can't predict probabilities in unsupervised "
                             "network")


class LazyLearning(object):
    """ Mixin for Lazy learning Neural Network algorithms.
    """
    def __init__(self, *args, **kwargs):
        self.input_train = None
        self.target_train = None
        super(LazyLearning, self).__init__(*args, **kwargs)

    def setup_defaults(self):
        del self.shuffle_data

    def train(self, input_train, target_train):
        self.input_train = input_train
        self.target_train = target_train

        if input_train.shape[0] != target_train.shape[0]:
            raise ValueError("Input data size must be the same as "
                             "target data size")

    def train_epoch(self, *args, **kwargs):
        raise AttributeError("This network doesn't have train epochs")

    def predict(self, input_data):
        if self.input_train is None:
            raise ValueError("Train your network before make prediction")
