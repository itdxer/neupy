Signals
=======

Basics
~~~~~~

Signal is a simple approach that helps you implement functions which will be triggered in specific case.
In training process there are 2 of them.
First one will be triggered after epoch training procedure.
Also you can controll at the rate in which network will trigger this signal.
The second one will be triggered when all traning is finished.

By default all signals are already defined and they display useful information about training in ``verbose`` mode.

Here is the example where we define two custom signals which will display useful information.

.. code-block:: python

    from neupy import algorithms

    def on_epoch_end(network):
        print(network.last_epoch)

    def on_training_end(network):
        print("Training finished")

    bpnet = algorithms.GradientDescent(
        (2, 5, 1),
        epoch_end_signal=on_epoch_end,
        train_end_signal=on_training_end,
        show_epoch=100,
        verbose=True
    )

We define two new functions that rewrite all outputs and display their own information on every trigger signal.
Also we define parameter ``show_epoch`` that will control ``epoch_end_signal`` signal frequency.
It's useful when there are a lot of epochs in network and we don't want to see each of them.

Property ``verbose`` just control logging output, but doesn't make any effect on signals.

Early stopping
~~~~~~~~~~~~~~

The other useful feature releated to the signals is that you can can implement your own rule that can interrupt training procedure. Here is a simple example:

.. code-block:: python

    from neupy import algorithms
    from neupy.algorithms import StopTrainingException

    def on_epoch_end(network):
        if network.errors.last() < 0.001:
            raise StopTrainingException("Stop training")

    gdnet = algorithms.GradientDescent(
        (2, 3, 1),
        epoch_end_signal=on_epoch_end,
    )
