import numpy as np


__all__ = ('iter_until_converge',)


def iter_until_converge(network, epsilon, max_epochs):
    """
    Train network until error converged or maximum number of
    epochs has been reached.

    Parameters
    ----------
    network : BaseNetwork instance

    epsilon : float
        Interrupt training in case if different absolute
        between two previous errors is less than specified
        epsilon value.

    max_epochs : int
        Maximum number of epochs to train.
    """
    logs = network.logs

    # Trigger first iteration and store first error term
    yield network.last_epoch

    previous_error = error_delta = network.training_errors[-1]
    epoch = network.last_epoch

    while error_delta > epsilon:
        epoch = epoch + 1
        network.last_epoch += 1

        yield epoch

        last_error = network.training_errors[-1]
        error_delta = abs(last_error - previous_error)
        previous_error = last_error

        if epoch >= max_epochs and error_delta > epsilon:
            logs.message(
                "TRAIN", (
                    "Epoch #{} interrupted. Network didn't converge "
                    "after {} iterations".format(epoch, max_epochs)
                )
            )
            return

    if np.isnan(error_delta) or np.isinf(error_delta):
        logs.message(
            "TRAIN", (
                "Epoch #{} interrupted. "
                "Delta between errors NaN or Inf.".format(epoch)))
    else:
        logs.message(
            "TRAIN", "Epoch #{} interrupted. Network converged.".format(epoch))
