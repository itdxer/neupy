.. _discrete-hopfield-network:

Discrete Hopfiel Network
========================

.. contents::

At this tutorial we are going to understand :network:`Discrete Hopfield Network <DiscreteHopfieldNetwork>` algorithm.

Architecture
------------

:network:`Discrete Hopfield Network <DiscreteHopfieldNetwork>` is a very simple algorithm.
To understend it you must to know how to make an outer product and product between matrix and vector.
That's it.
Network include just these two operations.

`Discrete` means that network can save and predict only the binary vectors.
For this specific network we will use only sign-binary numbers: 1 and -1.
We can't use zeros, because it can reduce information from the input vector, later in this tutorial I try explain why that is the problem.

By now, let's check how we can train and use :network:`Discrete Hopfield Network <DiscreteHopfieldNetwork>`.

Training procedure
~~~~~~~~~~~~~~~~~~

First step as usual is the training procedure.
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

Where :math:`W` is a weight matrix and :math:`x` is an input vector.
:math:`x_i` can only be -1 or 1 values, so after outer product matrix always has 1 on the diagonal.
This values on the diagonal are useles, so we better to remove them.
For this reason we need to set up all the diagonal values equal to the zero, just to make sure that we don't have any information on the diagonals that can make incorrect contribution into the output result.
The final weight formula look like this:

.. math::

    \begin{align*}
        W =
        \left[
        \begin{array}{c}
          0 & x_1 x_2 & \cdots & x_1 x_n \\
          x_2 x_1 & 0 & \cdots & x_2 x_n \\
          \vdots\\
          x_n x_1 & x_n x_2 & \cdots & 0 \\
        \end{array}
        \right]
    \end{align*}

If we already had some information stored in the weights, we just add the new weight matrix to the old one.

.. math::

    W = W_{old} + W_{new}

But if you need to store multiple vectors inside the network at the same time you don't need to compute weight for each vector and than sum up them.
If you have a matrix :math:`X \in \Bbb R^{m\times n}` where each row is the input vector, then you can just make product between it transpose and itself.

.. math::

    W = X^T X - m I


Where :math:`I` is an identity matrix (:math:`I \in \Bbb R^{n\times n}`), :math:`n` is a number of features in the input vector and :math:`m` is a number of input patterns inside the matrix :math:`X`.

Recovery from memory
~~~~~~~~~~~~~~~~~~~~

Recovery operation is also very simple.
If you have a vector that contains broken information and you need to recovery it, you can just multiply the input vector by the weight matrix.

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

.. math::

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

And before use the network output we must filter it throw the :math:`sign` function.

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
With 1 and -1 values we don't lose information after dot product operation, we just collect everything inside the weight matrix.
But if we had zeros, we would remove all information that stored in the weight column even is value associated with zero was correct.
You can use 0 and 1 values and sometime you will get the corect result, but this situation typicaly would be much rare that for the 1 and -1 values.

Memory limit
------------

Obviously, you can't store infinite number of vectors inside the network.
There already exists a good rule of thumb.
Suppose that :math:`n` is the dimention of your input vector, then the formula below compute the upper limit for the number of input vectors that you are able to save inside the :network:`Discrete Hopfield Network <DiscreteHopfieldNetwork>`.

.. math::

    l = \left \lfloor \frac{n}{2 \cdot log(n)} \right \rfloor

Formula above doesn't mean that you can't save more values than :math:`l`.
It is just a good upper bound for typical tasks, but you can find some situations when this rule will fail.

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

Basicly outer product just repeat vector 4 times with excaly the same value or with inversed signs.
Look closer to the matrix :math:`U` that we got.
First and third column (or row, it doesn't metter, because matrix is symmetric) are exacly the same as input vector.
The second and fourth are also the same, but with the opposite sign.
That beause in the vector :math:`u` we have 1 on the first and third places and -1 on the rest.

To make weight from the :math:`U` matrix, we need to remove ones from the diagonal to make them valid.

.. math::

    W = U - I

:math:`I` is the identity matrix and :math:`I \in \Bbb R^{n \times n}`, where :math:`n` is a number of features in the input vector.

When we have one stored vector inside the weights we don't realy need to remove ones from the diagonal.
The main problem would be when we have more than one vector stored in the weights.
Each value on the diagonal would be equal to the number of stored vectors inside of it.
On recovery procedure these diagonal elements will produce the big values for the output vector and eventually they will impair the output result.

Hallucinations
--------------

Hallucinations is one of the possible problem in the :network:`Discrete Hopfield Network <DiscreteHopfieldNetwork>`.
Sometimes network output produce something that we didn't teach it.

To understand this phenomenon we must first of all define the Hopfield energy function.

.. math::

    E = -\frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} w_{ij} x_i x_j - \sum_{i=1}^{n} \theta_i x_i

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

But linear algebra notation works only with the :math:`x` vector, we can't use matrix :math:`X` with the multiple input patterns instead of the :math:`x` in this formula.

Summary
-------

.. author:: default
.. categories:: none
.. tags:: memory, unsupervised
.. comments::
