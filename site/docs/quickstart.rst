Quick start
===========

Ready to get started?

MNIST problem
*************

MNIST problem is probably the most known for those who have already heared about neural networks.
The most popular neural network algorithm probably is :network:`GradientDescent`.
Let's try to solve the MNIST problem using :network:`GradientDescent`
First of all we need to load data.

.. code-block:: python

    >>> from sklearn import datasets, cross_validation
    >>> mnist = datasets.fetch_mldata('MNIST original')  
    >>> data, target = mnist.data, mnist.target

I used scikit-learn to fetch the MNIST dataset, but you can do that in
different way.

Data doesn't have appropriate for for neural network, so we need to make simple
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

Next we need to divide ataset into two parts: train and test. Regarding `The
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
 
Let's start with architecture. I didn't reinvent the wheel and use one of the
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
    ...         layers.Relu(400),
    ...         layers.Dropout(0.2),
    ...         layers.Softmax(400),
    ...         layers.ArgmaxOutput(10),
    ...     ],
    ...     error='categorical_crossentropy',
    ...     step=0.01,
    ...     verbose=True,
    ...     shuffle_data=True,
    ...     momentum=0.99,
    ...     nesterov=True,
    ... )


Now we are going to define :network:`GradientDescent` neural network which solves this problem.
First of all we have to set up basic structure for network and add some useful configurations.
As problem is nonlinear we should add one hidden layer to the network.
For first network implementation we have to set up number of hidden units inside network randomly.
Let the units number be 4.

.. code-block:: python

    >>> from neupy import algorithms
    >>> bpnet = algorithms.GradientDescent(
    ...     (2, 4, 1),
    ...     step=0.1,
    ...     verbose=True,
    ...     show_epoch='4 times',
    ... )

As you can see from code additionaly we set up ``step`` and ``show_epoch`` parameters.
``step`` parameter control learning rate.
``show_epoch`` controls the frequency display in the terminal training.
We set the value up to ``'4 times'`` that mean we will see network progress 4 times and one additional for the final iteration.

We set up network connections as tuple of layers sizes, but we don't put in activation function.
That is because :network:`GradientDescent` use the most common sigmoid layer by
default for tuple structure.
More about layer configuration you can read `here <layers.html>`_.

If you run the code in terminal you will see output which looks like this one:

.. image:: ../_static/screenshots/bpnet-config-logs.png
    :width: 70%
    :align: center
    :alt: GradientDescent configuration output

From this output we can extract a lot of information about network configurations.

First of all, as we can see, most of options have gray color label, but
some of them are green.
Green color defines all options which we put in network manually and gray color options are default parameters.
This output shows all possible properties neural network configurations.
All properties separeted on few groups and each group is a :network:`GradientDescent`  parent classes.
More information about :network:`GradientDescent` algorithm properties you will find in documentation, just click on algorithm name link and you will see it.

Now we are going to train network to solve the XOR problem.
Let set up ``5000`` epochs for training procedure and check the result.

.. code-block:: python

    >>> bpnet.train(input_data, target_data, epochs=5000)

Output in terminal should look similar to this one:

.. image:: ../_static/screenshots/bpnet-train-logs.png
    :width: 70%
    :align: center
    :alt: GradientDescent training procedure output

In the output you can see many useful information about learning procedures.
First of all there is simple information about input data and number of training epochs.
Also ther you can see information about every 1000 training epoch.
In addition training output always shows the last training epoch.
Each epoch output has three values: Train error, Validation error and Epoch time.
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
    :alt: GradientDescent epoch errors plot

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
