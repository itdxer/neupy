from __future__ import division

from numpy import where, exp, log, cosh, tanh as np_tanh, clip

from neupy.functions import with_derivative


__all__ = ('sigmoid', 'linear', 'step', 'tanh', 'rectifier', 'softplus',
           'softmax')


# -----------------------------------------------------#
#             Non-deferentiable functions              #
# -----------------------------------------------------#


def step(x):
    return where(x > 0, 1, 0)


def rectifier(x):
    return where(x < 0, 0, x)


# -----------------------------------------------------#
#              Functions with derivatives              #
# -----------------------------------------------------#


def linear_deriv(x):
    return 1


@with_derivative(linear_deriv)
def linear(x):
    return x


def softmax_deriv(x, temp=1):
    softmax_result = softmax(x, temp)
    return softmax_result * (1 - softmax_result)


@with_derivative(softmax_deriv)
def softmax(x, temp=1):
    exp_input = exp(x / temp)
    exp_sum = exp_input.sum(axis=1)
    return exp_input / exp_sum.reshape((x.shape[0], 1))


# -----------------------------------------------------#
#        Functions with second-order derivatives       #
# -----------------------------------------------------#


def second_softplus_deriv(x):
    exp_for_x = exp(x)
    return exp_for_x / (1 + exp_for_x) ** 2


@with_derivative(second_softplus_deriv)
def softplus_deriv(x):
    exp_for_x = exp(x)
    return exp_for_x / (1 + exp_for_x)


@with_derivative(softplus_deriv)
def softplus(x):
    return log(1 + exp(x))


def second_sigmoid_deriv(x, alpha=1):
    exp_value = exp(alpha * x)
    return (
        -alpha ** 2 * (exp_value * (exp_value - 1))
    ) / (
        (exp_value + 1) ** 3
    )


@with_derivative(second_sigmoid_deriv)
def sigmoid_deriv(x, alpha=1):
    sigmoig_output = sigmoid(x, alpha=alpha)
    return alpha * sigmoig_output * (1 - sigmoig_output)


@with_derivative(sigmoid_deriv)
def sigmoid(x, alpha=1.):
    output = 1 / (1 + exp(-alpha * x))
    # Fix for infinite results
    return clip(output, a_min=0, a_max=1)


def tanh_second_deriv(x, alpha=1):
    return -2 * alpha ** 2 * tanh(x, alpha) * (1 / cosh(alpha * x)) ** 2


@with_derivative(tanh_second_deriv)
def tanh_deriv(x, alpha=1):
    return alpha - alpha * tanh(x, alpha) ** 2


@with_derivative(tanh_deriv)
def tanh(x, alpha=1):
    return np_tanh(alpha * x)
