Signals and Early stopping
==========================

Basics
------

The simplest signal can be defined as a function. This function will be triggered after every epoch.

.. code-block:: python

    from neupy import algorithms
    from neupy.layers import *

    def on_epoch_end(optimizer):
        print("Last epoch: {}".format(optimizer.last_epoch))

    optimizer = algorithms.GradientDescent(
        Input(3) >> Sigmoid(2) >> Sigmoid(1),
        signals=on_epoch_end,
        show_epoch=100,
        verbose=True
    )

Signals that must intercept multiple events during the training can be specified as a class where each method should have the same name as the event that it has to intercept.

.. code-block:: python

    class PrintEventSignal(object):
        def train_start(self, optimizer, **kwargs):
            print('Triggers when training starts')

        def epoch_start(self, optimizer):
            print('Triggers when epoch starts')

        def update_start(self, optimizer):
            print('Triggers when parameter update started')

        def train_error(self, optimizer, value, **kwargs):
            training_loss = value
            print('Triggers when training error has been calculated')

        def valid_error(self, optimizer, value, **kwargs):
            validation_loss = value
            print('Triggers when validation error has been calculated')

        def update_end(self, optimizer):
            print('Triggers when parameters were updated')

        def epoch_end(self, optimizer):
            print('Triggers when training epoch finished')

        def train_end(self, optimizer):
            print('Triggers when training finished')

    optimizer = algorithms.GradientDescent(
        Input(3) >> Sigmoid(2) >> Sigmoid(1),
        signals=PrintEventSignal(),
        show_epoch=100,
        verbose=True
    )

Multiple signals can be used during the training.

.. code-block:: python

    optimizer = algorithms.GradientDescent(
        Input(3) >> Sigmoid(2) >> Sigmoid(1),
        signals=[
            # Mix of functions and class instances
            on_epoch_end,
            PrintEventSignal(),
            PrintEventSignal(),
        ],
        show_epoch=100,
        verbose=True
    )

Early stopping
--------------

Signals allow us to interrupt training process.

.. code-block:: python

    from neupy import algorithms, layers
    from neupy.exceptions import StopTraining

    def on_epoch_end(optimizer):
        if optimizer.errors.valid[-1] < 0.001:
            raise StopTraining("Training has been interrupted")

    optimizer = algorithms.GradientDescent(
        [
            layers.Input(784),
            layers.Relu(500),
            layers.Relu(300),
            layers.Softmax(10),
        ],
        signals=on_epoch_end,
    )

If we use constructible architectures than it's possible to save parameter after each training epoch and load them in case if validation error increases.

.. code-block:: python

    from neupy import algorithms, layers, storage
    from neupy.exceptions import StopTraining

    def on_epoch_end(optimizer):
        epoch = optimizer.last_epoch
        errors = optimizer.errors.valid

        if len(errors) >= 2:
            if errors[-1] > errors[-2]:
                # Load parameters and stop training
                storage.load(optimizer, 'training-epoch-{}.pickle'.format(epoch - 1))
                raise StopTraining("Training has been interrupted")
            else:
                # Save parameters after successful epoch
                storage.save(optimizer, 'training-epoch-{}.pickle'.format(epoch))

    optimizer = algorithms.GradientDescent(
        [
            layers.Input(784),
            layers.Relu(500),
            layers.Relu(300),
            layers.Softmax(10),
        ],
        signals=on_epoch_end,
    )
