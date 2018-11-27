.. _mnist-classification:

MNIST Classification
====================

.. raw:: html

    <div class="short-description">
        <p>
        This short tutorial shows how to design and train simple network for digit classification in NeuPy.
        </p>
    </div>


.. image:: images/random-digits.png
    :align: center
    :alt: MNIST digits example

This short tutorial shows how to build and train simple network for digit classification in NeuPy.

Data preparation
----------------

Data can be loaded in different ways. I used scikit-learn to fetch the MNIST dataset.

.. code-block:: python

    >>> from sklearn import datasets
    >>> mnist = datasets.fetch_mldata('MNIST original')
    >>> X, y = mnist.data, mnist.target

Now that we have the data we need to confirm that we have expected number of samples.

.. code-block:: python

    >>> X.shape
    (70000, 784)
    >>> y.shape
    (70000,)

Every data sample has 784 features and can be reshaped into 28x28 image.

.. code-block:: python

    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(X[0].reshape((28, 28)), cmap='gray')
    >>> plt.show()

.. image:: images/digit-example.png
    :width: 70%
    :align: center
    :alt: MNIST digit example

In this tutorial, we will use each image as a vector so we won't need to reshape it to its original size. The only thing that we need to do is to rescale image values. Rescaling image will help network to converge faster.

.. code-block:: python

    >>> X /= 255.
    >>> X -= X.mean(axis=0)

Notice the way division and subtraction are specified. In this way, we make update directly on the ``X`` matrix without copying it. It can be validated with simple example.

.. code-block:: python

    >>> A = np.random.random((100, 10))
    >>> id(A)  # numbers will be different between runs
    4486892960
    >>>
    >>> A -= 3
    >>> id(A)  # object ID didn't change
    4486892960
    >>>
    >>> A = A - 3
    >>> id(A)  # and now it's different, because it's different object
    4602409968

After last update for matrix ``A`` we got different identifier for the object, which means that it got copied.

In case of the in-place updates, we don't waste memory. Current dataset is relatively small and there is no memory deficiency, but for larger datasets it might make a big difference.

There is one more processing step that we need to do before we can train our network. Let's take a look into target classes.

.. code-block:: python

    >>> import random
    >>> random.sample(y.astype('int').tolist(), 10)
    [9, 0, 9, 7, 2, 2, 3, 0, 0, 8]

All number that we have are specified as integers. For our problem we want network to learn visual representation of the numbers. We cannot use them as integers, because it will create some problems during the training. Basically, with this definition we're implying that number ``1`` visually more similar to ``0`` than to number ``7``. It happens only because difference between ``1`` and ``0`` smaller than difference between ``1`` and ``7``. In order to avoid making any type of assumptions we will use one-hot encoding technique.

.. code-block:: python

    >>> from sklearn.preprocessing import OneHotEncoder
    >>> encoder = OneHotEncoder(sparse=False)
    >>> y = encoder.fit_transform(y.reshape(-1, 1))
    >>> y.shape
    (70000, 10)

You can see that every digit was transformed into a 10 dimensional vector.

And finally, we need to divide our data into training and validation set. We won't show validation set to the network and we will use it only to test network classification accuracy.

.. code-block:: python

    >>> import numpy as np
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> x_train, x_test, y_train, y_test = train_test_split(
    ...     X.astype(np.float32),
    ...     y.astype(np.float32),
    ...     test_size=(1 / 7.)
    ... )

Notice that data was converted into 32 bit float numbers. This is the only float type that currently supported by NeuPy.

Model initialization
--------------------

Networks architecture and training algorithm can be defined in a single statement.

.. code-block:: python

    >>> from neupy import algorithms, layers
    >>>
    >>> network = algorithms.Momentum(
    ...     [
    ...         layers.Input(784),
    ...         layers.Relu(500),
    ...         layers.Relu(300),
    ...         layers.Softmax(10),
    ...     ],
    ...     error='categorical_crossentropy',
    ...     step=0.01,
    ...     verbose=True,
    ...     shuffle_data=True,
    ...     momentum=0.99,
    ...     nesterov=True,
    ... )

Isn't it simple and clear? All the most important information related to the neural network you can find in the terminal output. If you run the code that shown above you would get the same output as on the figure below.

.. image:: images/bpnet-config-logs.png
    :width: 70%
    :align: center
    :alt: Gradient Descent configuration

From this output we can extract a lot of information about network configurations.

First of all, as we can see, most of options have green color label, but some of them are gray. Green color defines all options which we put in network manually and gray color options are default parameters. All properties separeted on few groups and each group is a :network:`Momentum`  parent classes. More information about :network:`Momentum` algorithm properties you will find in documentation, just click on algorithm name link and you will see it.

In addition for feedforward neural networks it's possible to check architecture in form of a table.

.. code-block:: python

    >>> network.architecture()

.. image:: images/bpnet-architecture.png
    :width: 70%
    :align: center
    :alt: Neural Network Architecture

Training
--------

Now we are going to train network. Let set up 20 epochs for training procedure and check the result.

.. code-block:: python

    >>> network.train(x_train, y_train, x_test, y_test, epochs=20)

Output in terminal should look similar to this one:

.. image:: images/bpnet-train-logs.png
    :width: 70%
    :align: center
    :alt: GradientDescent training procedure output

Output show the most important information related to training procedure. Each epoch contains 4 columns. First one identified epoch. The second one show training error. The third one is optional. In case you have validation dataset, you can check learning perfomanse using dataset separated from the learning procedure. And the last column shows how many time network trains during this epoch.

Evaluations
-----------

From the table is not clear network's training progress. We can check it very easy. Network instance contains built-in method that build line plot that show training progress. Let's check our progress.

.. code-block:: python

    >>> from neupy import plots
    >>> plots.error_plot(network)

.. image:: images/bpnet-train-errors-plot.png
    :width: 70%
    :align: center
    :alt: GradientDescent epoch errors plot

From the figure above you can notice that validation error does not decrease over time. Sometimes it goes up and sometimes down, but it doesn't mean that network trains poorly. Let's check small example that can make this problem clear.

.. code-block:: python

    >>> actual_values = np.array([1, 1, 1])
    >>> model1_prediction = np.array([0.9, 0.9, 0.4])
    >>> model2_prediction = np.array([0.6, 0.6, 0.6])

In the code above you can see two prediction releate to the different models. The first model predicted two samples right and one wrong. The second one predicted everything right. But second model's predictions are less certain. Let's check the cross entropy error.

.. code-block:: python

    >>> from neupy import estimators
    >>> estimators.binary_crossentropy(actual_values, model1_prediction)
    0.3756706118583679
    >>> estimators.binary_crossentropy(actual_values, model2_prediction)
    0.5108255743980408

That is the result that we looked for. The second model made better prediction, but it got a higher cross entropy error. It means that we less certain about our prediction. Similar situation we've observed in the plot above.

Let's finally make a simple report for our classification result.

.. code-block:: python

    >>> from sklearn import metrics
    >>>
    >>> y_predicted = network.predict(x_test).argmax(axis=1)
    >>> y_test = np.asarray(y_test.argmax(axis=1)).reshape(len(y_test))
    >>>
    >>> print(metrics.classification_report(y_test, y_predicted))
            precision    recall  f1-score   support

        0       0.98      0.99      0.99       936
        1       0.99      0.99      0.99      1163
        2       0.98      0.98      0.98       982
        3       0.98      0.99      0.98      1038
        4       0.98      0.98      0.98       948
        5       0.99      0.98      0.98       921
        6       0.99      0.99      0.99      1013
        7       0.98      0.98      0.98      1029
        8       0.98      0.98      0.98       978
        9       0.98      0.96      0.97       992

        avg / total       0.98      0.98      0.98     10000

    >>> score = metrics.accuracy_score(y_test, y_predicted)
    >>> print("Validation accuracy: {:.2%}".format(score))
    Validation accuracy: 98.37%

The 98.37% accuracy is pretty good for such a quick solution. Additional modification can improve prediction accuracy.


.. author:: default
.. categories:: none
.. tags:: classification, tutorials, supervised, backpropagation, image recognition, deep learning
.. comments::
