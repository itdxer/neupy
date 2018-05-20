Environments
============

Reproducible environment
------------------------

This functionality can help you make your code reproducible.

.. code-block:: python

    from neupy import environment
    environment.reproducible()

The ``reproducible`` function accepts only one optional argument ``seed``. By default, it is equal to ``0``.

Speedup calculation
-------------------

Set up 32-bit float type for all tensors and scallers. In addition, this method disables garbage collector and speed up the calculation, but it increases an amount of memory that Tensorflow uses for computation.

.. code-block:: python

    from neupy import environment
    environment.speedup()
