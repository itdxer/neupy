Signals and Early stopping
==========================

Basics
------

Signal is a special type of functions that can be triggered after a certain event. Most of the NeuPy algorithm supports two types of events

1. ``epoch_end_signal`` - event that triggers after each training epoch
2. ``train_end_signal`` - event that triggers at the end of the training

.. code-block:: python

    from neupy import algorithms

    def on_epoch_end(bpnet):
        print("Last epoch: {}".format(bpnet.last_epoch))

    def on_training_end(bpnet):
        print("Training finished")

    bpnet = algorithms.GradientDescent(
        (2, 5, 1),
        epoch_end_signal=on_epoch_end,
        train_end_signal=on_training_end,
        show_epoch=100,
        verbose=True
    )

Each signal can be defined as a function that accepts one mandatory argument. When event callback function triggers training algorithm passes its instance as a first argument to the callback.

Early stopping
--------------

Signals allow us to interrupt training process.

.. code-block:: python

    from neupy import algorithms, layers
    from neupy.exceptions import StopTraining

    def on_epoch_end(gdnet):
        if gdnet.validation_errors[-1] < 0.001:
            raise StopTraining("Training has been interrupted")

    gdnet = algorithms.GradientDescent(
        [
            layers.Input(784),
            layers.Relu(500),
            layers.Relu(300),
            layers.Softmax(10),
        ],
        epoch_end_signal=on_epoch_end,
    )

If we use constructible architectures than it's possible to save parameter after each training epoch and load them in case if validation error increases.

.. code-block:: python

    from neupy import algorithms, layers, storage
    from neupy.exceptions import StopTraining

    def on_epoch_end(gdnet):
        epoch = gdnet.last_epoch
        errors = gdnet.validation_errors

        if len(errors) >= 2:
            if errors[-1] > errors[-2]:
                # Load parameters and stop training
                storage.load(gdnet, 'training-epoch-{}.pickle'.format(epoch - 1))
                raise StopTraining("Training has been interrupted")
            else:
                # Save parameters after successful epoch
                storage.save(gdnet, 'training-epoch-{}.pickle'.format(epoch))

    gdnet = algorithms.GradientDescent(
        [
            layers.Input(784),
            layers.Relu(500),
            layers.Relu(300),
            layers.Softmax(10),
        ],
        epoch_end_signal=on_epoch_end,
    )
