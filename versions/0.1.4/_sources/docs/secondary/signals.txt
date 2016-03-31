Signals
=======

Signal is a simple approach that helps you implement functions which will be triggered in specific case.
In training process there are 2 of them.
First one will be triggered after epoch training procedure.
Also you can controll at the rate in which network will trigger this signal.
The second one will be triggered when all traning is finished.

By default all signals are already defined and they display useful information about training in ``verbose`` mode.

Here is the example where we define two custom signals which will display useful information.

.. code-block:: python

    from neupy import algorithms

    def train_epoch_end(network):
        network.logs.data("""
            Epoch {}
            Step: {}
        """.format(network.epoch, network.step))

    def train_end(network):
        network.logs.log("TRAIN", "End train")

    bpnet = algorithms.ConjugateGradient(
        (2, 5, 1),
        train_epoch_end_signal=train_epoch_end,
        train_end_signal=train_epoch_end,
        show_epoch=100,
        verbose=True
    )

We define two new functions that rewrite all outputs and display their own information on every trigger signal.
Also we define parameter ``show_epoch`` that will control ``train_epoch_end_signal`` signal frequency.
It's useful when there are a lot of epochs in network and we don't want to see each of them.

Property ``verbose`` just control logging output, but doesn't make any effect on signals.
