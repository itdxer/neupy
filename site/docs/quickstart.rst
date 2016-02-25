Quick start
===========

Ready to get started?

MNIST classification
********************

The MNIST problem is probably the most known for those who have already 
heared about neural networks. This short tutorial contains simple solution for this
problem that you can quickly build by your own using NeuPy. Let's dive in.

First of all we need to load data.

.. code-block:: python

    >>> from sklearn import datasets, cross_validation
    >>> mnist = datasets.fetch_mldata('MNIST original')  
    >>> data, target = mnist.data, mnist.target

I used scikit-learn to fetch the MNIST dataset, but you can do that in
different way.

Data doesn't have appropriate format for neural network, so we need to make simple
transformation before apply it to neural network.

.. code-block:: python

    >>> from sklearn.preprocessing import OneHotEncoder
    >>> 
    >>> data = data / 255.
    >>> data = data - data.mean(axis=0) 
    >>>
    >>> target_scaler = OneHotEncoder()
    >>> target = target_scaler.fit_transform(target.reshape((-1, 1))
    >>> target = target.todense()

Next we need to divide dataset into two parts: train and test. Regarding `The
MNIST Database <http://yann.lecun.com/exdb/mnist/>`_ page we wil use 60,000
samples for training and 10,000 for test.

.. code-block:: python

    >>> from sklearn.cross_validation import train_test_split 
    >>> x_train, x_test, y_train, y_test = train_test_split(
    ,,,     data.astype(np.float32),
    ...     target.astype(np.float32),
    ...     train_size=(6. / 7)
    ... )

In the previous procedure I converted all data to `float32` data type. This
simple trick will help us use less memory and decrease computation time.
Theano is a main backend for the Gradient Descent based algorithms in NeuPy.
For Theano we need add additional configuration that will explain Theano that
we are going to use 32bit float numbers.

.. code-block:: python

    >>> import theano
    >>> theano.config.floatX = 'float32'

We prepared everything that we need for neural network training. Now we are
able to create neural network that will classify digits for us. 
 
Let's start with an architecture. I didn't reinvent the wheel and used one of the
know architectures from `The MNIST Database
<http://yann.lecun.com/exdb/mnist/>`_ page which is 784 > 500 > 300 > 10. As
the main activation function I used Relu and Softmax for the final layer. The
main algorithm is Nesterov Momentum that use 100 samples per batch iteration.
Actually all this and other network configuration should be clear from the code.

.. code-block:: python

    >>> network = algorithms.Momentum(
    ...     [
    ...         layers.Relu(784),
    ...         layers.Dropout(0.2),
    ...         layers.Relu(500),
    ...         layers.Dropout(0.2),
    ...         layers.Softmax(300),
    ...         layers.ArgmaxOutput(10),
    ...     ],
    ...     error='categorical_crossentropy',
    ...     step=0.01,
    ...     verbose=True,
    ...     shuffle_data=True,
    ...     momentum=0.99,
    ...     nesterov=True,
    ... )

Isn't it simple and clear? All the most important information related to the neural
network you can find in the terminal output. If you run code that shown above
you would get the same output as on the figure below.

.. image:: ../_static/screenshots/bpnet-config-logs.png
    :width: 70%
    :align: center
    :alt: GradientDescent configuration output

From this output we can extract a lot of information about network configurations.

First of all, as we can see, most of options have green color label, but
some of them are gray.
Green color defines all options which we put in network manually and gray
color options are default parameters.
All properties separeted on few groups and each group is a :network:`Momentum`  parent classes.
More information about :network:`Momentum` algorithm properties you will 
find in documentation, just click on algorithm name link and you will see it.

Now we are going to train network.
Let set up ``20`` epochs for training procedure and check the result.

.. code-block:: python

    >>> network.train(x_train y_train, x_test, y_test, epochs=20)

Output in terminal should look similar to this one:

.. image:: ../_static/screenshots/bpnet-train-logs.png
    :width: 70%
    :align: center
    :alt: GradientDescent training procedure output

Output show the most important information related to training procedure.
Each epoch contains 4 columns. First one identified epoch. 
The second one show training error. The third one is optional.
In case you have validation dataset, you can check learning perfomanse using
dataset separated from the learning procedure.
And the last column shows how many time network trains during this epoch.

From the table is not clear network's trainig progress. We can check it very easy.
Network instance contains built-in method that build line plot that show trainig progress.
Let's check our progress.

.. code-block:: python

    >>> network.plot_errors()

.. image:: ../_static/screenshots/bpnet-train-errors-plot.png
    :width: 70%
    :align: center
    :alt: GradientDescent epoch errors plot

Let's make a simple report for our classification result.

.. code-block:: python

    >>> y_predicted = network.predict(x_test)
    >>> y_test = np,asarray(y_test.argmax(axis=1)).reshape(len(y_test))
    >>>
    >>> print(metrics.classification_report(y_test, y_predicted))
    precision
    >>> score = metrics.accuracy_score(y_test, y_predicted)
    >>> print("Validation accuracy: {:.2f}%".format(100 * score))
    Validation accuracy: 98.31%

The 98.3% accuracy is pretty good for such a quick solution. Additional modification can 
improve prediction accuracy.
