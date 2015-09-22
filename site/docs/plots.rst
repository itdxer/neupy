Plots
=====

Hinton diagram
--------------

More information about the :plot:`hinton` function you can find in documentation.

.. code-block:: python

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from neupy import plots
    >>>
    >>> weight = np.random.randn(20, 20)
    >>>
    >>> plt.style.use('ggplot')
    >>> plt.figure(figsize=(16, 12))
    >>> plt.title("Hinton diagram")
    >>> plots.hinton(weight)
    >>> plt.show()

.. figure:: images/plots-hinton-example.png
    :width: 100%
    :align: center
    :alt: Hinton diagram example from NeuPy library
