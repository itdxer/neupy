import theano
import theano.sparse
import theano.tensor as T
from neupy import layers
from neupy.utils import asint


class AverageLinearLayer(layers.Layer):
    def output(self, input_value):
        input_value = asint(input_value)
        summator = self.weight[input_value]
        self.parameters = [self.weight]
        return summator.mean(axis=1)


def crossentropy(expected, predicted, epsilon=1e-10):
    predicted = T.clip(predicted, epsilon, 1.0 - epsilon)
    n_samples = expected.shape[0]

    error = theano.sparse.sp_sum(-expected * T.log(predicted))
    return error / n_samples


# Make it possible to use sparse matrix for training target
crossentropy.expected_dtype = theano.sparse.csr_matrix


def accuracy_score(expected, predicted):
    return (predicted == expected).sum() / len(predicted)
