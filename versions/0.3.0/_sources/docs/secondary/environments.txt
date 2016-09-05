Environments
============

This is a simple functionality that can help you set up your environment.

Reproducible environment
------------------------

This functionality can help you make your code reproducible.

.. code-block:: python

    from neupy import environment
    environment.reproducible()

The ``reproducible`` function accepts only one optional argument ``seed``. By default it is equal to ``0``.

Sandbox environment
-------------------

This functionality helps set up some Theano configurations that can speed up compilation time and slow down computation time. It is useful for experimentation.

.. code-block:: python

    from neupy import environment
    environment.sandbox()
