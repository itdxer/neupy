.. _discrete-hopfield-network:

Discrete Hopfiel Network
========================

.. contents::

At this tutorial we are going to understand :network:`Discrete Hopfield Network <DiscreteHopfieldNetwork>` algorithm.

Architecture
------------

For this task we will use :network:`Discrete Hopfield Network <DiscreteHopfieldNetwork>`.
It is the simplest Neural Networks architectures.
There are two things which you are need to know, how to make the outer and matrix-vector products.
That's it.
Network include just this two operations.

`Discrete` means that network can save and predict only binary vectors.
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

:math:`x_i` can only be -1 or 1 values, so after outer product matrix always has 1 on the diagonal.
Information on the diagomal is useles, so we better to remove it.
For this reason we set up all diagonal values equal to zero.
The final weight dormula will look like this:

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

And finally, if we already had some information stored in weights we just add the new weight matrix to the old one.

.. math::

    W = W_{old} + W_{new}

But if you need to store multiple vectors inside the network at the same time you don't need to compute weight for every vector and than sum up them.
For example, if you have a matrix :math:`X` where each row is an input vector, then you can just make product between it transpose and itself.

.. math::

    W = X^T X

Recovery from memory
~~~~~~~~~~~~~~~~~~~~

Recovery operation is also very simple.
If you have the vector which contains broken information and you nedd to recover it, you can just multiple it by the weight matrix.

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
            &-1 && : x < 1
        \end{array}
    \right.\\

    y = sign(s)

That's it.
Now :math:`y` store the recovered vector :math:`x`.

Maybe now you can see why we can't use zeros in the vector input vectors.
With 1 and -1 values we don't lose information in the dot product operation, we just collect everything.
But if we had zeros we would remove all information which is store in the weight column.
In my experiense you can use it and sometime you will get the corect result, but this situation typicaly would be much rare that for the 1 and -1 values.

Problem
~~~~~~~

Obviously, you can't store infinite number of vectors inside the network.
There already exists a good rule of thumb.
Suppose that :math:`n` is the dimention of your input vector, then the formula below compute the upper limit for the number of input vectors that you are able to save inside the :network:`Discrete Hopfield Network <DiscreteHopfieldNetwork>`.

.. math::

    \left \lfloor \frac{n}{2 \cdot log(n)} \right \rfloor

Formula above doesn't mean that you can't save more values that this formula output produce.
It is just a good upper bound for typical tasks, but you can find some situations when this rule will fail.

Why does it work?
-----------------

Let's start with the simple vector :math:`u`.

.. math::

    u = \left[\begin{align*}1 \\ -1 \\ 1 \\ -1\end{align*}\right]

Assume that the network don't have patterns inside of it, so the vector :math:`u` would be the first one.
Let's compute weights for the network.

.. math::

    \begin{align*}
        W = u u^T =
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

Basicly outer product just repeat vector 4 times with different scale.
Values are always equal to 1 or -1, so we get the same vector or just the same with reversed sign.
Look closer to the matrix :math:`W` that we got.
First and third column (or row, it doesn't metter) are exacly the asme as input vector.
The second and fourth are also the same, but with the opposite sign.

And remove ones from the diagonal to make weights valid.

.. math::

    W = u u^T - I

Where :math:`I` is the identity matrix.

When we have one stored vector inside the weights we don't realy need to remove ones from the diagonal.
The main problem would be when we have many vectors stored in the weights.
Each value on the diagonal would be equal to the number of stored vectors inside of it.
On recovery procedure this diagonal elements will produce a big weight for the output vector, but in fact they don't contain an important information.

Hallucinations
--------------

Hallucinations is one of the possible problem in :network:`Discrete Hopfield Network <DiscreteHopfieldNetwork>`.
Not sure that it is 100% correct term, but I think about it like that.
Sometimes network output produce something that we didn't teach it.

To understand it we must first of all define the Energy Function.

.. math::

    E = \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} w_{ij} v_i v_j + \sum_{i=1}^{n} \theta_i v_i

Summary
-------

.. author:: default
.. categories:: none
.. tags:: memory, unsupervised
.. comments::
