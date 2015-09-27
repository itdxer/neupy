__all__ = ("docs",)


docs = {
    # ------------------------------------ #
    #                Methods               #
    # ------------------------------------ #

    "last_error": """last_error()
        Returns the last error network result after training procedure
        or ``None`` value if you try to get it before network training.
    """,
    "plot_errors": """plot_errors(use_semilog=False)
        Draws the error rate update plot. It always shows network
        learning progress. When you add cross validation data set
        into training function it displays validation data set error as
        separated curve. If parameter ``use_semilog`` is equal to the
        ``True`` value it displays x-axis in logarithmic scale.
    """,
    "predict": """predict(input_data)
        Predict value.
    """,
    "fit": """fit(\*args, \*\*kwargs)
        The same as ``train`` method.
    """,

    # ------------------------------------ #
    #             Train Methods            #
    # ------------------------------------ #

    "supervised_train": """train(input_train, target_train, input_test=None,\
    target_test=None, epochs=100, epsilon=None):
        Trains network. You can control network training procedure
        iterations with the number of epochs or converge value epsilon.
        Also you can specify ``input_test`` and ``target_test`` and control
        your validation data error on each iteration.
    """,
    "supervised_train_epochs": """train(input_data, target_data, epochs=100):
        Trains network with fixed number of epochs.
    """,
    "unsupervised_train_epochs": """train(input_train, epochs=100):
        Trains network with fixed number of epochs.
    """,
    "unsupervised_train_epsilon": """train(input_train, epsilon=1e-5):
        Trains network until it is converged.
    """,
    "supervised_train_lazy": """train(input_train, target_train, copy=True):
        Network just stores all the information about the data and use it for \
        the prediction. Parameter ``copy`` copy input data before store it \
        inside the network.
    """,

    # ------------------------------------ #
    #              Parameters              #
    # ------------------------------------ #

    "verbose": """verbose : bool
        Verbose output for network. Defaults to ``True``. ``False``
        disables output.
    """,
    "step": """step : float
        Learns step, defaults to ``0.1``.
    """,
    "show_epoch": """show_epoch : int
        This option controls how offen you'd see the epoch updates
        information in the terminal, defaults to ``1``
    """,
    "shuffle_data": """shuffle_data : bool
        If it's ``True`` class shuffles all your training data before
        training your network, defaults to ``True``.
    """,
    "error": """error : function
        Function which controls your training error. defaults to ``mse``
    """,
    "use_bias": """use_bias : bool
        Uses bias in the network, defualts to ``True``.
    """,
    "train_epoch_end_signal": """train_epoch_end_signal : function
        Calls this function when train epoch finishes.
    """,
    "train_end_signal": """train_end_signal : function
        Calls this function when train process finishes.
    """,

    # ------------------------------------ #
    #                 Steps                #
    # ------------------------------------ #
    "first_step": """first_step : float
        Contains initialized step value.
    """,
    "steps": """steps : list of float
        List of steps in the same order as the network layers.
        By default all values are equal to ``step`` parameter.
    """,

    # ------------------------------------ #
    #               Warnings               #
    # ------------------------------------ #
    "bp_depending": """It works with any algorithm based on backpropagation. \
    Class can't work without it.
    """
}


# ------------------------------------ #
#         Complex parameters           #
# ------------------------------------ #


def joindocs(docs, docskeys):
    return ''.join([docs[key] for key in docskeys])


full_params_params = ()
docs.update({
    'full_params': joindocs(
        docs,
        [
            'step', 'show_epoch', 'shuffle_data',
            'error', 'use_bias', 'train_epoch_end_signal',
            'train_end_signal', 'verbose'
        ]
    ),
    'full_signals': joindocs(
        docs, ['train_epoch_end_signal', 'train_end_signal']
    ),
    'full_methods': joindocs(
        docs, ['fit', 'predict', 'last_error', 'plot_errors']
    )
})
