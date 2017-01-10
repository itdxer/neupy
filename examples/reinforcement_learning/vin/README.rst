Value Iteration Network (VIN)
=============================

Simple implementation of Value Iteration Network (VIN) in NeuPy. Original code for the paper you can find `here <https://github.com/avivt/VIN>`_.

Network trains on Gridworld dataset from the VIN paper.

Code description
----------------

.. csv-table::
    :header: "File name", "Description"

    "loaddata.py","Read data from MAT file, split it into train and test samples and store everything in pickle files"
    "vin.py","Train VIN and validate its accuracy"
    "visualize.py","Sample a few grids from test samples and visualize trajectory predicted by the pretrained network"

Usage
-----

To train network you need to run following command

.. code-block:: bash

    $ python vin.py

Visualizations
--------------

8x8 Grid world
~~~~~~~~~~~~~~

.. image:: images/8x8-gridworld-trajectories.png
    :width: 50%
    :align: center
