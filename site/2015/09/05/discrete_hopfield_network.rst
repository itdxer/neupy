.. _discrete-hopfield-network:

Discrete Hopfiel Network
========================

.. contents::

At this tutorial we are going to understand :network:`Discrete Hopfield Network <DiscreteHopfieldNetwork>` algorithm.

:network:`Discrete Hopfield Network <DiscreteHopfieldNetwork>` is a type of algorithms which called - `Autoassociative memories <https://en.wikipedia.org/wiki/Autoassociative_memory>`_
Don't afraid of word `Autoassociative`.
The idea behind this type of algorithms is simple.
They can store in `memory` useful information and later they are able to reproduce it from the partialy broken patterns.
You can think about of it as a human memory.
For instance, imagine that you see an old picture with a place where you was a long time ago, but this picture is in a very bad quality and with a high blur.
From the picture you can reconize some objects or places that are very familiar to you.
It can be a house or a lake or anything else which can clearly create an associations for you.
With this details other parts of picture begin to make more sense.
Even if you don't clearly see all objects you can recover a big part of picture from your memory just from the few familiar details.
That what it is all about.
Autoassociative memory network is a way to think about the human memory and try to build something useful that have a similar behaviour.

On this tutorial you will see only intuition behind the network.
I avoid all proofs to make this tutorial more simple.
If you are interesting in proofs of the :network:`Discrete Hopfield Network <DiscreteHopfieldNetwork>` you can check them at R. Rojas. Neural Networks [1]_ book.

Architecture
------------

:network:`Discrete Hopfield Network <DiscreteHopfieldNetwork>` is a simple algorithm.
It simple because you don't need to much background to use it.
Everything you need to know is just how to make a basic Linear Algebra operations, like outer product or sum of the two matrices.

Let's start with a simple thing.
What do we know about it so far?
Just a name and a type.
From the name we can identify one useful thing about the network.
It's `Discrete`.
That mean that network can works only with the binary vectors.
But for this network we wouldn't use the binary numbers in a tupical form.
Instead of them we will use the bipolar numbers.
They are almost the same, but except the 0 we will use -1 to decode a negative state.
We can't use zeros.
There are two main reasons for it.
First one is that zeros reduce the information from the network weight, later in this tutorial we will see it.
The second one is more complex, it depence on the neature of bipolar vectors.
Basically they are more likely to be othogonal to each other which is a critical moment for the :network:`Discrete Hopfield Network <DiscreteHopfieldNetwork>`.
But as I said before we wouldn't talk about proofs and anything out of basic understanding of the Linear Algebra operations.

So, let's check how we can train and use the :network:`Discrete Hopfield Network <DiscreteHopfieldNetwork>`.

Training procedure
~~~~~~~~~~~~~~~~~~

We can't use the memory without any patterns stored in it.
So first of all we are going to understand how to train the network.
For the :network:`Discrete Hopfield Network <DiscreteHopfieldNetwork>` train procedure doesn't require any iterations.
It includes just an outer product between input vector and it transpose.

.. math::

    \begin{align*}
        W = x \cdot x^T =
        \left[
        \begin{array}{c}
          x_1\\
          x_2\\
          \vdots\\
          x_n
        \end{array}
        \right]
        \cdot
        \left[
        \begin{array}{c}
          x_1 & x_2 & \cdots & x_n
        \end{array}
        \right]
    \end{align*}
    =

.. math::

    \begin{align*}
        =
        \left[
        \begin{array}{c}
          x_1^2 & x_1 x_2 & \cdots & x_1 x_n \\
          x_2 x_1 & x_2^2 & \cdots & x_2 x_n \\
          \vdots\\
          x_n x_1 & x_n x_2 & \cdots & x_n^2 \\
        \end{array}
        \right]
    \end{align*}

:math:`W` is a weight matrix and :math:`x` is an input vector.
Each value :math:`x_i` from the input vector can be only -1 or 1.
So on the diagonal we have only squared values and that mean we will always see ones on that places.
Think about it, at any case values on the diagonal take just one possible state.
In recovery procedure they are perceived as a definite memory state, but in fact they are useles and even can make incorrect contribution into the output result.
For this reason we need to set up all the diagonal values equal to the zero.
The final weight formula look like the one below.

.. math::

    \begin{align*}
        W =
        x x^T - I =
        \left[
        \begin{array}{c}
          0 & x_1 x_2 & \cdots & x_1 x_n \\
          x_2 x_1 & 0 & \cdots & x_2 x_n \\
          \vdots\\
          x_n x_1 & x_n x_2 & \cdots & 0 \\
        \end{array}
        \right]
    \end{align*}

Where :math:`I` is an identity matrix.

But usualy we need to save more values in the memory.
For another pattern we must make exacly the same procedure as before and than just add the generated weight matrix to the old one.

.. math::

    W = W_{old} + W_{new}

And this procedure generate us a new weight that would be valid for the both stored patterns.
Later you can add another patterns using the same algorithm.

But if you need store the multiple vectors inside the network at the same time you don't need to compute the weight for the each vector and than sum up them.
If you have a matrix :math:`X \in \Bbb R^{m\times n}` where each row is the input vector, then you can just make product between it transpose and itself.

.. math::

    W = X^T X - m I


Where :math:`I` is an identity matrix (:math:`I \in \Bbb R^{n\times n}`), :math:`n` is a number of features in the input vector and :math:`m` is a number of input patterns inside the matrix :math:`X`.
Ofcourse you must remove all values on the diagonal and in math term it's better to show with the difference between weight and identity.
In practice it's not very good store a big identity matrix (if dimention is really big) just to set up zeros on the diagonal.
Usualy linear algebra libraries give you a possibility to set up diagonal without additional matrix and this solution would be more efficient.
For example in NumPy library it's a `numpy.fill_diagonal <http://docs.scipy.org/doc/numpy/reference/generated/numpy.fill_diagonal.html>`_ function

Recovery from memory
~~~~~~~~~~~~~~~~~~~~

Recovery procedure include pattern recovery from the broken one.
There are already exists two main approaches. First one recover the vector using each value from the vector one time. The second one is a randomized approach. The basic idea is that you iteratively get random value from the input vector and recover it from the network. I will explain each approach in more details.

Synchronous recovery approach
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Full recovery approach is much easier so we are going to check it first.
To recover your pattern from the memory you can just multiply the weight matrix by the input vector.

.. math::

    \begin{align*}
        s = {W}\cdot{x}=
        \left[
        \begin{array}{cccc}
          w_{11} & w_{12} & \ldots & w_{1n}\\
          w_{21} & w_{22} & \ldots & w_{2n}\\
          \vdots & \vdots & \ddots & \vdots\\
          w_{n1} & w_{n2} & \ldots & w_{nn}
        \end{array}
        \right]
        \left[
        \begin{array}{c}
          x_1\\
          x_2\\
          \vdots\\
          x_n
        \end{array}
        \right]
        =
    \end{align*}

    \begin{align*}
        =
        \left[
            \begin{array}{c}
              w_{11}x_1+w_{12}x_2 + \cdots + w_{1n} x_n\\
              w_{21}x_1+w_{22}x_2 + \cdots + w_{2n} x_n\\
              \vdots\\
              w_{n1}x_1+w_{n2}x_2 + \cdots + w_{nn} x_n\\
            \end{array}
        \right]
    \end{align*}


Variable :math:`s` doesn't contain recover pattern.
As you can see we sum up all information from the weights.
It's clear that value not necessary equal to -1 or 1, so we must do something else with this output and make the result values as bipolar numbers.

Let's think about this product operation.
What does it actualy do?
Basically after outer product we save our pattern dublicated :math:`n` times (where :math:`n` is a number of features in input vector) inside the weight (we will see it later in this tutorial).
When we store more patterns we get interception between them (its called a **crosstalk**) and each pattern add some noise to the another patterns.
So, after product between :math:`W` and :math:`x` for each value from the vector :math:`x` we get a recovered vector with a little bit noise.
For :math:`x_1` we get a first column from the matrix :math:`W`, for the :math:`x_2` a second column, and so on.
Pay attention that we reverse sign before store it if :math:`x_i = -1` and in recovery operation we reverse it back, so after recovery procedure sign would be valid.
Next we add all vectors together.
This operation looks like voting.
For example we have 3 vectors.
If the first two vectors have 1 at first position and the third one has -1 at the same position, so the winner must be value 1.
We can make the same voting procedure with :math:`sign` function.
So the output value must be 1 if total value is greater then zero and -1 otherwise.

.. math::

    sign(x) = \left\{
        \begin{array}{lr}
            &1 && : x \ge 0\\
            &-1 && : x < 0
        \end{array}
    \right.\\

    y = sign(s)

That's it.
Now :math:`y` store the recovered vector :math:`x`.

Maybe now you can see why we can't use zeros in the input vectors.
In `voting` procedure we use each row with the bipolar vectors, but if values were zeros they will ignore column from the weight matrix.

Ofcourse you can use 0 and 1 values and sometime you will get the correct result, but this approach would be worse than with the bipolar values.

Asynchronous recovery approach
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Previous approach is good, but it has limitations.
If you change one value in input vector it can change your output result.
The another popular approach is a randomization.
You randomly select a value from your input vector and associat it with a column from the weight matrix.
You repeat this procedure multiple times and after some number of iterations you just sum up all vectors that you are already select.
In terms of the :network:`Discrete Hopfield Network <DiscreteHopfieldNetwork>` we can say that **neuron fired**

Let's check the example:
Suppouse we already have a weight matrix :math:`W` with one pattern inside.

.. math::

    \begin{align*}
        W =
        \left[
        \begin{array}{cccc}
          0 & 1 & -1 \\
          1 & 0 & -1 \\
          -1 & -1 & 0
        \end{array}
        \right]
    \end{align*}

Let's assume that we have a vector :math:`x` from which we want to recover the pattern.

.. math::

    \begin{align*}
        x =
        \left[
            \begin{array}{c}
              1\\
              -1\\
              -1
            \end{array}
        \right]
    \end{align*}

At the first iteration we randomly chose a value.
Let it be the first one.
So we multiple the first column by this selected value.

.. math::

    \begin{align*}
        y_1 =
        1 \cdot \left[
            \begin{array}{c}
              -1\\
              -1\\
              0
            \end{array}
        \right] =
        \left[
            \begin{array}{c}
              0\\
              1\\
              -1
            \end{array}
        \right]
    \end{align*}

At the second iteration we again chose the random value.
Now we get the third one and we again repeat the same precodure

.. math::

    \begin{align*}
        y_2 =
        -1 \cdot \left[
            \begin{array}{c}
              -1\\
              -1\\
              0
            \end{array}
        \right] =
        \left[
            \begin{array}{c}
              1\\
              1\\
              0
            \end{array}
        \right]
    \end{align*}

We can repeat these operation many times, but at the end we always sum up all values :math:`y_i`.
Just after the two iterations network output must be like the one below.

.. math::

    \begin{align*}
        y = sign(y_1 + y_2) =
        sign(
            \left[
                \begin{array}{c}
                  0\\
                  1\\
                  -1
                \end{array}
            \right] +
            \left[
                \begin{array}{c}
                  1\\
                  1\\
                  0
                \end{array}
            \right]
        ) =
        sign(
            \left[
                \begin{array}{c}
                  1\\
                  2\\
                  -1
                \end{array}
            \right]
        ) =
        \left[
            \begin{array}{c}
              1\\
              1\\
              -1
            \end{array}
        \right]
    \end{align*}


Memory limit
------------

Obviously, you can't store infinite number of vectors inside the network.
There already exists two good rule of thumbs.

Suppose that :math:`n` is the dimention (number of features) of your input vector and :math:`m` is the number of patterns that you want to save inside the network.

The first rule gives a simple ration between :math:`m` and :math:`n`.

.. math::

    m \approx 0.18 n

THe main problem with this rule is that proof assume that stored vectors inside the weight are completly random with an equaly probability.
Unfortunately that is not always right.
Suppose we save the images of numbers from 0 to 9.
On pictures colors are black and white, so we can encode them in bipolar vectors.
Let's assume that vectors can be random.
But will the probabilities equal of obtaining each value?
Usualy not.
More likely that number of white pixels would be greater than number of black.
Before use this rule you must think about type of your input patterns.

The second formula gives another ration.
Proportion is logarithmically.

.. math::

    m = \left \lfloor \frac{n}{2 \cdot log(n)} \right \rfloor

It doesn't mean that you can't save more values than :math:`m`.
It is just a good upper bound for typical tasks, but you can find some situations when this rule will fail.

Hallucinations
--------------

Hallucinations is one of the possible problem in the :network:`Discrete Hopfield Network <DiscreteHopfieldNetwork>`.
Sometimes network output produce something that we didn't teach it.

To understand this phenomenon we must first of all define the Hopfield energy function.

.. math::

    E = -\frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} w_{ij} x_i x_j + \sum_{i=1}^{n} \theta_i x_i

Where :math:`w_{ij}` is a weight value on the :math:`i`-th row and :math:`j`-th column.
:math:`x_i` is a :math:`i`-th values from the input vector :math:`x`.
:math:`\theta` is a threshold.
For the :network:`Discrete Hopfield Network <DiscreteHopfieldNetwork>` we can assume that :math:`\theta` equal to 0.
For :network:`Discrete Hopfield Network <DiscreteHopfieldNetwork>` the energy function looks little bit simpler.

.. math::

    E = -\frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} w_{ij} x_i x_j

In terms of a linear algebra we can write formula for the Energy Function more simplier.

.. math::

    E = -\frac{1}{2} x^T W x

But linear algebra notation works only with the :math:`x` vector, we can't use matrix :math:`X` with the multiple input patterns instead of the :math:`x` in this formula, beause after product your energies would be on the diagonal and the other values would be useles.

Why does it work?
-----------------

Let's start with an example.
Suppose we have a vector :math:`u`.

.. math::

    u = \left[\begin{align*}1 \\ -1 \\ 1 \\ -1\end{align*}\right]

Assume that network don't have patterns inside of it, so the vector :math:`u` would be the first one.
Let's compute weights for the network.

.. math::

    \begin{align*}
        U = u u^T =
        \left[
            \begin{array}{c}
                1 \\
                -1 \\
                1 \\
                -1
            \end{array}
        \right]
        \left[
            \begin{array}{c}
                1 & -1 & 1 & -1
            \end{array}
        \right]
        =
        \left[
            \begin{array}{cccc}
                1 & -1 & 1 & -1\\
                -1 & 1 & -1 & 1\\
                1 & -1 & 1 & -1\\
                -1 & 1 & -1 & 1
            \end{array}
        \right]
    \end{align*}

Look closer to the matrix :math:`U` that we got.
Outer product just repeat vector 4 times with the same or inversed value.
First and third column (or row, it doesn't metter, because matrix is symmetric) are exacly the same as input vector.
The second and fourth are also the same, but with the opposite sign.
That beause in the vector :math:`u` we have 1 on the first and third places and -1 on the rest.

To make weight from the :math:`U` matrix, we need to remove ones from the diagonal.

.. math::

    W = U - I

:math:`I` is the identity matrix and :math:`I \in \Bbb R^{n \times n}`, where :math:`n` is a number of features in the input vector.

When we have one stored vector inside the weights we don't realy need to remove ones from the diagonal.
The main problem would be when we have more than one vector stored in the weights.
Each value on the diagonal would be equal to the number of stored vectors inside of it.
On recovery procedure these diagonal elements will produce the big values for the output vector and eventually they will impair the output result.

Example
-------

Let's define few images that we are going to teach the network.

.. code-block:: python

    >>> import numpy as np
    >>> from neupy import algorithms
    >>>
    >>> def draw_bin_image(image_matrix):
    ...     for row in image_matrix.tolist():
    ...         print('| ' + ' '.join(' *'[val] for val in row))
    ...
    >>> zero = np.matrix([
    ...     0, 1, 1, 1, 0,
    ...     1, 0, 0, 0, 1,
    ...     1, 0, 0, 0, 1,
    ...     1, 0, 0, 0, 1,
    ...     1, 0, 0, 0, 1,
    ...     0, 1, 1, 1, 0
    ... ])
    >>>
    >>> one = np.matrix([
    ...     0, 1, 1, 0, 0,
    ...     0, 0, 1, 0, 0,
    ...     0, 0, 1, 0, 0,
    ...     0, 0, 1, 0, 0,
    ...     0, 0, 1, 0, 0,
    ...     0, 0, 1, 0, 0
    ... ])
    >>>
    >>> two = np.matrix([
    ...     1, 1, 1, 0, 0,
    ...     0, 0, 0, 1, 0,
    ...     0, 0, 0, 1, 0,
    ...     0, 1, 1, 0, 0,
    ...     1, 0, 0, 0, 0,
    ...     1, 1, 1, 1, 1,
    ... ])
    >>>
    >>> draw_bin_image(zero.reshape((6, 5)))
    |   * * *
    | *       *
    | *       *
    | *       *
    | *       *
    |   * * *

We have 3 images, so now we can train network with these patterns.

.. code-block:: python

    >>> data = np.concatenate([zero, one, two], axis=0)
    >>>
    >>> dhnet = algorithms.DiscreteHopfieldNetwork()
    >>> dhnet.train(data)

That's all.
Now to make sure that network catch patterns we can introduce the broken pattern.

.. code-block:: python

    >>> half_zero = np.matrix([
    ...     0, 1, 1, 1, 0,
    ...     1, 0, 0, 0, 1,
    ...     1, 0, 0, 0, 1,
    ...     0, 0, 0, 0, 0,
    ...     0, 0, 0, 0, 0,
    ...     0, 0, 0, 0, 0,
    ... ])
    >>> draw_bin_image(half_zero.reshape((6, 5)))
    |   * * *
    | *       *
    | *       *
    |
    |
    |
    >>>
    >>> half_two = np.matrix([
    ...     0, 0, 0, 0, 0,
    ...     0, 0, 0, 0, 0,
    ...     0, 0, 0, 0, 0,
    ...     0, 1, 1, 0, 0,
    ...     1, 0, 0, 0, 0,
    ...     1, 1, 1, 1, 1,
    ... ])
    >>> draw_bin_image(half_two.reshape((6, 5)))
    |
    |
    |
    |   * *
    | *
    | * * * * *

We define the same image, but without the lower half of it.
Now we can reconstruct pattern from the memory.

.. code-block:: python

    >>> result = dhnet.predict(half_zero)
    >>> draw_bin_image(result.reshape((6, 5)))
    |   * * *
    | *       *
    | *       *
    | *       *
    | *       *
    |   * * *
    >>>
    >>> result = dhnet.predict(half_two)
    >>> draw_bin_image(result.reshape((6, 5)))
    | * * *
    |       *
    |       *
    |   * *
    | *
    | * * * * *

Cool!
Network catch the pattern right.

From this network we also can catch the hallucination.
We need to define another pattern and again try to recover it.

.. code-block:: python

    >>> half_two = np.matrix([
    ...     1, 1, 1, 0, 0,
    ...     0, 0, 0, 1, 0,
    ...     0, 0, 0, 1, 0,
    ...     0, 0, 0, 0, 0,
    ...     0, 0, 0, 0, 0,
    ...     0, 0, 0, 0, 0,
    ... ])
    >>>
    >>> result = dhnet.predict(half_two)
    >>> draw_bin_image(result.reshape((6, 5)))
    |   * *
    |     *
    |     *
    |   * *
    | *   *
    | * * * * *

We didn't clearly teach the network for this pattern.
But if we look closer, it looks like mixed patter of numbers 1 and 2.
That is exacly hallucination.
Basically network create new local minimum some where between numbers 1 and 2 that looks very close two both but still non of them.

For the specific input network produce the same output.
There exists another aproach where we randomly select some of the input patterns and try to mix them.
Somethimes this approach works very well.
For this specific example you are able to catch the valid output of number 2, but this event would be rare.
You can test it by your own.

.. code-block:: python

    >>> dhnet = algorithms.DiscreteHopfieldNetwork(
    ...     mode='async',
    ...     n_times=400
    ... )
    >>>
    >>> dhnet.train(data)
    >>> result = dhnet.predict(half_two)
    >>> draw_bin_image(result.reshape((6, 5)))
    | * * *
    |     *
    |     *
    |   * *
    | *   *
    | * * * * *
    >>> result = dhnet.predict(half_two)
    >>> draw_bin_image(result.reshape((6, 5)))
    | * * *
    |       *
    |       *
    |   * *
    | *
    | * * * * *

I catched it from the second time, but sometimes it takes more iterations.
Usualy to improve the accuracy of this method you can define more number of iterations for the random procedure (``n_times`` parameter).
Another way is to add additional verification and repeat it if output patter fail expectation.

.. figure:: images/hopfield-energy-vis.png
    :width: 80%
    :align: center
    :alt: Asynchronous Discrete Hopfield Network energy update after each iteration

And finally we can look closer on the network memory using Hinton diagram

.. code-block:: python

    >>> from neupy import plots
    >>> import matplotlib.pyplot as plt
    >>>
    >>> plt.figure(figsize=(14, 12))
    >>> plt.title("Hinton diagram")
    >>> plots.hinton(dhnet.weight)
    >>> plt.show()

.. figure:: images/hinton-diagram.png
    :width: 80%
    :align: center
    :alt: Asynchronous Discrete Hopfield Network energy update after each iteration

Summary
-------

The :network:`Discrete Hopfield Network <DiscreteHopfieldNetwork>` is a very simple and you need a little knowlege in linear algebra to understand it.

Also you can check another ':ref:`Password recovery <password-recovery>`' tutorial in which the password is recovered from the memory of the :network:`Discrete Hopfield Network <DiscreteHopfieldNetwork>`.

References
----------

.. [1] \R. Rojas. Neural Networks. In Associative Networks. pp. 311 - 336, 1996.

.. [2] Math4IQB. (2013, November 17). Hopfield Networks. Retrieved
     from https://www.youtube.com/watch?v=gfPUWwBkXZY

.. [3] \R. Callan. The Essence of Neural Networks. In Pattern Association. pp. 84 - 98, 1999.

.. author:: default
.. categories:: none
.. tags:: memory, unsupervised
.. comments::
