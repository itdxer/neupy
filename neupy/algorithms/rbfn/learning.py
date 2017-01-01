from neupy.core.docs import SharedDocs
from neupy.core.properties import WithdrawProperty


__all__ = ('LazyLearningMixin',)


class LazyLearningMixin(SharedDocs):
    """
    Mixin for lazy learning Neural Network algorithms.

    Notes
    -----
    - Network uses lazy learning which mean that network doesn't
      need iterative training. It just stores parameters
      and use them to make a predictions.

    Methods
    -------
    train(input_train, target_train, copy=True)
        Network just stores all the information about the data and use
        it for the prediction. Parameter ``copy`` copies input data
        before saving it inside the network.
    """
    step = WithdrawProperty()
    show_epoch = WithdrawProperty()
    shuffle_data = WithdrawProperty()
    train_end_signal = WithdrawProperty()
    epoch_end_signal = WithdrawProperty()

    def __init__(self, *args, **kwargs):
        self.input_train = None
        self.target_train = None
        super(LazyLearningMixin, self).__init__(*args, **kwargs)

    def train(self, input_train, target_train):
        self.input_train = input_train
        self.target_train = target_train

        if input_train.shape[0] != target_train.shape[0]:
            raise ValueError("Number of samples in the input and target "
                             "datasets are different")
