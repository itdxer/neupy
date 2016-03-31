Notifications
=============

In NeuPy only Twilio SMS notifications are available, but you can use your
own implementation for anything you need.
So the basic idea is that you override signal, for example ``train_end_signal``, and define a notification inside the new function.

Before useing Twilio API you should first install a library.

.. code-block:: bash

    $ pip install twilio

There is the simple example of program which will send SMS when network training
process will be finished.

.. code-block:: python

    from neupy import algorithms
    from neupy.helpers.sms import twilio_sms

    send_sms = twilio_sms(
        account_id="MY_ACCOUNT_ID",
        token="SECRET_TOKEN",
        to_phone="+XXXXXXXXXX",
        from_phone="+XXXXXXXXXX",
        verbose=True
    )

    def on_train_end(network):
        last_error = network.last_error()
        send_sms("Train finished. Last error: {}".format(last_error))

    lmnet = algorithms.LevenbergMarquardt(
        (10, 40, 1),
        train_end_signal=on_train_end
    )

More information about signals you can read in `documentation <signals.html>`_.
