Quick start
===========

Ready to get started?

XOR problem
***********

XOR problem is probably the most known for those who have already heared about neural networks.
The most popular neural network algorithm probably is :network:`Backpropagation`.
Let's try to solve the XOR problem using :network:`Backpropagation`
First of all we need to define 4 data samples for XOR function.

.. code-block:: python

    >>> import numpy as np
    >>>
    >>> np.random.seed(0)
    >>>
    >>> input_data = np.array([
    ...     [0, 0],
    ...     [0, 1],
    ...     [1, 0],
    ...     [1, 1],
    ... ])
    >>> target_data = np.array([
    ...     [1],
    ...     [0],
    ...     [0],
    ...     [1],
    ... ])

Let's check the data.

.. code-block:: python

    >>> import matplotlib.pyplot as plt
    >>>
    >>> plt.style.use('ggplot')
    >>> plt.scatter(*input_data.T, c=target_data, s=100)
    >>> plt.show()

.. image:: ../_static/screenshots/quick-start-data-viz.png
    :width: 70%
    :align: center
    :alt: Backpropagation configuration output

As we can see from chart on the top, problem is clearly nonlinear.

Now we are going to define :network:`Backpropagation` neural network which solves this problem.
First of all we have to set up basic structure for network and add some useful configurations.
As problem is nonlinear we should add one hidden layer to the network.
For first network implementation we have to set up number of hidden units inside network randomly.
Let the units number be 4.

.. code-block:: python

    >>> from neupy import algorithms
    >>> bpnet = algorithms.Backpropagation(
    ...     (2, 4, 1),
    ...     step=0.1,
    ...     show_epoch=1000,
    ... )

As you can see from code additionaly we set up ``step`` and ``show_epoch`` parameters.
``step`` parameter control learning rate.
``show_epoch`` controls the frequency display in the terminal training.
If we set the value up to ``1000`` we will see network progress for every ``1000`` epoch.

We set up network connections as tuple of layers sizes, but we don't put in activation function.
That is because :network:`Backpropagation` use the most common sigmoid layer by
default for tuple structure.
More about layer configuration you can read `here <layers.html>`_.

If you run the code in terminal you will see output which looks like this one:

.. image:: ../_static/screenshots/bpnet-config-logs.png
    :width: 70%
    :align: center
    :alt: Backpropagation configuration output

From this output we can extract a lot of information about network configurations.

First of all, as we can see, most of options have gray color label, but
some of them are green.
Green color defines all options which we put in network manually and gray color options are default parameters.
This output shows all possible properties neural network configurations.
All properties separeted on few groups and each group is a :network:`Backpropagation`  parent classes.
More information about :network:`Backpropagation` algorithm properties you will find in documentation, just click on algorithm name link and you will see it.

Now we are going to train network to solve the XOR problem.
Let set up ``5000`` epochs for training procedure and check the result.

.. code-block:: python

    >>> bpnet.train(input_data, target_data, epochs=5000)

Output in terminal should look similar to this one:

.. image:: ../_static/screenshots/bpnet-train-logs.png
    :width: 70%
    :align: center
    :alt: Backpropagation training procedure output

In the output you can see many useful information about learning procedures.
First of all there is simple information about input data and number of training epochs.
Also ther you can see information about every 1000 training epoch.
In addition training output always shows the last training epoch.
Each epoch output has three values: Error in, Error out and Epoch time.
Epoch time shows for how long the process was active in the specific epoch.
There are also two types of errors.
First one displays error for your training dataset and second one for validation dataset.
Validation data sample is optional and we are not using it in this example, but we can put in ``train`` method separated data sample and track validation error.

Our MSE looks well. Now we can visualize our errors in a chart.

.. code-block:: python

    >>> bpnet.plot_errors()

.. image:: ../_static/screenshots/bpnet-train-errors-plot.png
    :width: 70%
    :align: center
    :alt: Backpropagation epoch errors plot

And finally examine the prediction answer

.. code-block:: python

    >>> predicted = bpnet.predict(input_data)
    >>> predicted
    array([[ 0.77293114],
           [ 0.28974524],
           [ 0.18620525],
           [ 0.74104605]])

Looks well.
Using more training epochs can make better prediction.
For final step we just round our network result for making it valid.

.. code-block:: python

    >>> predicted.round()
    array([[ 1.],
           [ 0.],
           [ 0.],
           [ 1.]])
