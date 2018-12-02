Environments
============

Reproducible environment
------------------------

This functionality can help you make your code reproducible.

.. code-block:: python

    from neupy import environment
    environment.reproducible()

The ``reproducible`` function accepts only one optional argument ``seed``. By default, it is equal to ``0``.
