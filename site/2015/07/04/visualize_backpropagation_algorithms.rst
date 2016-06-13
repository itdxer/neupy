Visualize Algorithms based on the Backpropagation
=================================================

.. contents::

In this article we will test different algorithms based on the backpropagation method, visualize them and try to figure out some important features from a plot that we will get.

Checking data
-------------

First of all we need to define simple dataset which contains 6 points with two features.

.. code-block:: python

    import numpy as np

    input_data = np.array([
        [0.9, 0.3],
        [0.5, 0.3],
        [0.2, 0.1],
        [0.7, 0.5],
        [0.1, 0.8],
        [0.1, 0.9],
    ])
    target_data = np.array([
        [1],
        [1],
        [1],
        [0],
        [0],
        [0],
    ])

So we can make a scatter plot and look closer at this dots.

.. code-block:: python

    import matplotlib.pyplot as plt

    plt.scatter(input_data[:, 0], input_data[:, 1], c=target_data, s=100)
    plt.show()

.. figure:: images/visualize_gd/bp-vis-scatter.png
    :width: 80%
    :align: center
    :alt: Dataset scatter plot

From the figure above we can clearly see that all dots are linearly separable and we are able to solve this problem with simple perceptron. But a goal of this article is to make clear visualization of learning process for different algorithm based on the backpropagation method, so the problem must be as simple as possible, because in other cases it will be complex to visualize.

So, as the problem is linear separable we can solve it without hidden layers in network. There are two features and two classes, so we can build network which will take 2 input values and 1 output. We need just two weights, so we can visualize them in contour plot.

Initialize contour
------------------

I wouldn't add all code related to the plots building. In case if you are interested in the all code you can check the main script `here <https://github.com/itdxer/neupy/blob/master/examples/mlp/gd_algorithms_visualization.py>`_.

.. image:: images/visualize_gd/raw-contour-plot.png
    :width: 80%
    :align: center
    :alt: Approximation function contour plot

The plot above shows error rate that depends on the network's weights. The best result corresponds to the smallest error value. The best weights combination for this problem should be near the lower right corner in the white area.

Next, we are going to look at 5 algorithms based on the Backpropagation. They are:

* Gradient descent
* Momentum
* RPROP
* iRPROP+
* Conjugate Gradient + Golden Search

Let's define start point for our algorithms. As we can see from the figure above the position (-4, -4) and the error for it would be approximetly 0.43, so we define default weights on this position.

This function will train the network until the error will be smaller than `0.125`. Every network starts at place with coordinates `(-4, -4)` and finishes near the point with the closest value to `0.125`. The final result will depend on the selected algorithm.

Visualize algorithms based on the Backpropagation
-------------------------------------------------

Gradient Descent
++++++++++++++++

Let's primarily check :network:`Gradient Descent <GradientDescent>`.

.. figure:: images/visualize_gd/bp-steps.png
    :width: 80%
    :align: center
    :alt: Weight update steps for the Gradient Descent

Gradient Descent got to the value close to 0.125 using 797 steps and this black curve are just tiny steps of gradient descent algorithm. We can zoom it and look closer.

.. figure:: images/visualize_gd/bp-steps-zoom.png
    :width: 80%
    :align: center
    :alt: Zoomed weight update steps for the Gradient Descent

Now we can see some information about gradient descent algorithm. All steps for gradient descent algorithm have approximately similar magnitude. Their direction doesn't vary because contours in the zoomed picture are parallel to each other and in it we can see that there is still a lot of steps to achieve the minimum. Also we can see that small vectors are perpendicular to the contour.

The problem is that the step size a very sensitive parameter for the gradient descent. In typical problem we won't be able to visualize the learning progress and we won't have an ability to see that our updates over the epochs are inefficient. For this result I've used step size equal to ``0.3``, but if we had increase it to ``10`` we would reach our goal in ``25`` steps. I haven't added any improvments to make a fair comparison between other algorithms in the summary chapter.

Momentum
++++++++

Now let's look at another very popular algorithm - :network:`Momentum`.

.. figure:: images/visualize_gd/momentum-steps.png
    :width: 80%
    :align: center
    :alt: Momentum steps

:network:`Momentum` got to the value close to 0.125 by 92 steps, which is more than 8 times fewer steps than for the gradient descent. The basic idea behind :network:`Momentum` algorithm is that it accumulates gradients from the previous epochs. It means that if the graient has the same direction after each epoch weight update vector magnitude will increase. But if the gradient stars changing its direction weight update vector magnitude will decrease. Check the the figure again. Imagine that you stands near the pool with a very smooth surface. Than you just throw a ball inside of the pool in a way that makes it roll over the surface. While it rolls down gravity force drags it down and it makes ball roll faster and faster. Let's get back to the :network:`Momentum` algorithm and try to find these properties in the plot.

.. figure:: images/visualize_gd/momentum-steps-zoom.png
    :width: 80%
    :align: center
    :alt: Momentum steps zoom on increasing weight update size

When we zoom the plot we can see that the direction for weight update vectors is almost the same and gradient's direction doesn't change after each epoch. At the end of the zoomed plot above vector is bigger than the first one on the same plot. Since we always want to move forward we just speed up in one direction.

Let's get back to the ball example. What happens if the ball reach the bottom of the pool? Would it stop? Of cource not. Ball gained enough speed to keep moving forward. So it will go up. The ball starts slowing down, because gravity force is continue pooling it towards the bottom of the pool. Let's try to find similar behviour in the same plot.

.. figure:: images/visualize_gd/momentum-steps-zoom-decrease.png
    :width: 80%
    :align: center
    :alt: Momentum steps zoom on decreasing weight update size

From the figure above it’s clear that weight update magnitude become smaller. Like a ball that slows down and changes its direction towards the minimum.

And finaly to make it even more intuitive you can check weight update trajectory in 3D plot. It looks much closer to the ball and pool analogy than previous plot.

.. figure:: images/visualize_gd/momentum-3d-trajectory.png
    :width: 80%
    :align: center
    :alt: Momentum 3D trajector

RPROP
+++++

:network:`Momentum` makes fewer steps to reach the specified minimum point, but we still can do better. Next algorithm that we are going to check is :network:`RPROP`.

.. figure:: images/visualize_gd/rprop-steps.png
    :width: 80%
    :align: center
    :alt: RPROP steps

This improvment looks impressive. Now we are able to see steps without zooming. We got almost the same value as before using just 20 steps, which is approximately 5 times fewer than :network:`Momentum` and approximately 40 times fewer than :network:`Gradient Descent <GradientDescent>`.

Now we are going to figure out what are the main features of :network:`RPROP` just by looking at the plot above. :network:`RPROP` has a unique step for each weight. There are just two steps for each weight in the input layer for this network. :network:`RPROP` will increase the step size if gradient don't change the sign compare to previous epoch, and it will decrease otherwise.

Let's check a few first weights updates.

.. figure:: images/visualize_gd/rprop-first-11-steps.png
    :width: 80%
    :align: center
    :alt: RPROP first 11 steps

From the figure above you can see that first 11 updates have the same direction, so both steps are increase after each iteration. For the first epoch steps are equal to the same value which we set up at network initialization step. On the every next iterations they have been increased by the same constant factor, so after six iteration they became bigger, but they are still equal because they move in one direction all the time.

Now let's check the next epochs from the figure below. On the 12th epoch gradient changed the direction, but steps are still the same. But we can clearly see that gradient changed the sign for the second weight. :network:`RPROP` updated the step after weight had updated, so the step for the second weight must be fewer for the 13th epoch.

.. figure:: images/visualize_gd/rprop-11th-to-14th-epochs.png
    :width: 80%
    :align: center
    :alt: RPROP from 11th to 14th steps

Now let's look at the 13th epoch. It shows us how gradient sign difference on the 12th epoch updated steps. Now the steps are not equal. From the picture above we can see that update on the second weight (y axis) is smaler than on the first weight (x axis).

On the 16th epoch gradient on y axis changed the sign again. Network decreased by constant factor and update for the second weight on the 17th epoch would be fewer than on the 16th.

To train your intuition you can check the other epochs updates and try to figure out how steps are dependent on the direction.

iRPROP+
+++++++

:network:`iRPROP+ <IRPROPPlus>` is almost the same algorithm as :network:`RPROP` except a small addition.

.. figure:: images/visualize_gd/irprop-plus-steps.png
    :width: 80%
    :align: center
    :alt: iRPROP+ steps

As in :network:`RPROP` algorithm :network:`iRPROP+ <IRPROPPlus>` make exacly the same first 11 steps.

Now let's look at the 12th step in the figure below.

.. figure:: images/visualize_gd/irprop-plus-second-part.png
    :width: 80%
    :align: center
    :alt: iRPROP+ second part

Second weight (on the y axis) didn't change the value. On the same epoch :network:`RPROP` changed the gradient comparing to the previous epoch and just decreased step value after weight update. Instead, :network:`iRPROP+ <IRPROPPlus>` disabled weight update for current epoch (set it up to `0`). And of course it also decreased the step for the second weight. Also you can find that vector for the 12th epoch that looks smaller than for the :network:`RPROP` algorithm, because we ignored the second weight update. If we check the x axis update size we will find that it has the same value as in :network:`RPROP` algorithm.

On 13th epoch network included again second weight into the update process, because compared to the previous epoch gradient didn't change its sign.

Next steps are doing the same job, but 15th epoch differs from others. There are a few updates which are related specifically to :network:`iRPROP+ <IRPROPPlus>`, but the most important we have not seen before. After weight update on the 15th epoch network error increased, so our update made our prediction worse. Now on the 16th epoch network tried to rollback vector update. It decreased steps on the 15th epoch and weight update didn't go to the same point after the rollback procedure, it just took opposite direction with a smaller step.

Conjugate Gradient and Golden Search
++++++++++++++++++++++++++++++++++++

Now let's look at :network:`Conjugate Gradient <ConjugateGradient>` with :network:`Golden Search <LinearSearch>`. Conjugate Gradient in Neural Network variation is a little bit different than in numerical optimizations notation and it doesn't guarantee converge into n-th steps (`n` means dimmention size for specific problem). Steps don't have a perfect size for :network:`Conjugate Gradient <ConjugateGradient>`, so :network:`Golden Search <LinearSearch>` is always a good choice for a step selection.

.. figure:: images/visualize_gd/conj-grad-and-gold-search-steps.png
    :width: 80%
    :align: center
    :alt: Conjugate Gradient with Golden Search steps

From the figure above we can see almost perfect step for the specific direction. S. Each of the fourth steps make a great choice for the step size. Of course it's not a great assumption. Golden Search is just trying to figure out the most perfect step size by using a simple search and trying multiple different choices. In any case it's doing a greate job.

Finally network made 4 steps. If we add the same :network:`Golden Search <LinearSearch>` algorithm to the classic Gradient Descent we will get to the minimum using just 2 steps. You can try it by your own.

Bring them all together
-----------------------

.. figure:: images/visualize_gd/all-algorithms-steps.png
    :width: 80%
    :align: center
    :alt: All algorithms steps

Summary
-------

.. csv-table:: Summary table
    :header: "Algorithm", "Number of epochs"

    Gradient Descent, 797
    Momentum, 92
    RPROP, 20
    iRPROP+, 17
    Conjugate Gradient + Golden Search, 4

.. figure:: images/visualize_gd/compare-number-of-epochs.png
    :width: 80%
    :align: center
    :alt: Compare number of epochs

There is no perfect algorithm for neural network that can solve all problems.
All of them have there own pros and cons.
Some of the algorithms can be memory or computationally expensive and you have to choose an algorithm depend on the task which you want to solve.

All code is available in `GitHub <https://github.com/itdxer/neupy/blob/master/examples/mlp/gd_algorithms_visualization.py>`_. You can play around the script and set up the different learning algorithms and hyperparameters. More algorithms you can find in NeuPy's :ref:`cheat-sheet`.

.. author:: default
.. categories:: none
.. tags:: supervised, backpropagation, visualization
.. comments::
