Value Iteration Network (VIN)
=============================

Implementation of Value Iteration Network (VIN) in NeuPy. Original code for the paper you can find `here <https://github.com/avivt/VIN>`_.

Code description
----------------

.. csv-table::
    :header: "File name", "Description"

    "loaddata.py","Read data from MAT file, split it into train and test samples and store everything in pickle files"
    "train_vin.py","Train VIN and validate its accuracy"
    "visualize.py","Sample a few grids from test dataset and visualize trajectories predicted by the pretrained network"

Data preprocessing
------------------

To be able to convert .mat files to format suitable for network training you need to run ``loaddata.py`` file. Files for the 8x8 grid have been already generated and stored in the ``data`` directory. Other files you need to generate with the following command.

.. code-block:: bash

    $ # For grid world with 16x16 images
    $ python loaddata.py --imsize=16
    $
    $ # For grid world with 28x28 images
    $ python loaddata.py --imsize=28

Network training
----------------

In the ``models`` folder you can find pretrained weights for VIN network. Some you can run ``visualize.py`` script and play with the final results. In case if you are interested in trying network with different parameters then you can train network from scratch.

.. code-block:: bash

    $ # For grid world with 8x8 images
    $ python train_vin.py --imsize=8
    $
    $ # For grid world with 16x16 images
    $ python train_vin.py --imsize=16
    $
    $ # For grid world with 28x28 images
    $ python train_vin.py --imsize=28

Visualizations
--------------

8x8 Grid world
~~~~~~~~~~~~~~

.. code-block:: bash

    $ python visualize.py --imsize=8

.. image:: images/8x8-gridworld-trajectories.png
    :width: 50%
    :align: center

16x16 Grid world
~~~~~~~~~~~~~~~~

.. code-block:: bash

    $ python visualize.py --imsize=16

.. image:: images/16x16-gridworld-trajectories.png
    :width: 50%
    :align: center


28x28 Grid world
~~~~~~~~~~~~~~~~

.. code-block:: bash

    $ python visualize.py --imsize=28

.. image:: images/28x28-gridworld-trajectories.png
    :width: 50%
    :align: center
