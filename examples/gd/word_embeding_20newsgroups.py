import os

import dill
import theano
import theano.sparse
import theano.tensor as T
import numpy as np
import scipy.sparse as sp
from gensim import corpora
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from neupy import algorithms, layers


class AverageLinearLayer(layers.Layer):
    def output(self, input_value):
        input_value = T.cast(input_value, 'int32')
        summator = self.weight[input_value]
        self.parameters = [self.weight]
        return summator.mean(axis=1)


def crossentropy(expected, predicted, epsilon=1e-10):
    predicted = T.clip(predicted, epsilon, 1.0 - epsilon)
    return (
        theano.sparse.sp_sum(-expected * T.log(predicted))
    ) / expected.shape[0]


def tokenize_texts(texts):
    stoplist = stopwords.words('english')
    return [[w for w in t.lower().split() if w not in stoplist] for t in texts]


def text_windows(texts, dictionary, window_size=2):
    token2id = dictionary.token2id
    train_data, target_data = [], []

    for text in texts:
        text = [token2id[word] for word in text if word in token2id]
        for i in range(window_size, len(text) - window_size):
            target_data.append(text[i])
            train_data.append(text[i - window_size:i] +
                              text[i + 1:i + window_size + 1])

    n_samples = len(target_data)
    dictionary_size = len(token2id)
    row_indeces = np.arange(n_samples)
    column_indeces = np.array(target_data)

    target_data = sp.csr_matrix(
        (
            np.ones(n_samples),
            (row_indeces, column_indeces),
        ),
        shape=(n_samples, dictionary_size),
        dtype=theano.config.floatX,
    )

    return np.array(train_data), target_data


CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

window_size = 5
minimized_space = 100

theano.config.floatX = 'float32'

categories = ['rec.autos']
train_newsgroups = fetch_20newsgroups(subset='all', categories=categories)
data = train_newsgroups.data

data = tokenize_texts(data)
dictionary = corpora.Dictionary(data, prune_at=100)
data, target = text_windows(data, dictionary, window_size)

dictionary_size = len(dictionary.token2id)

embeding_network = algorithms.Momentum(
    [
        AverageLinearLayer(dictionary_size),
        layers.Softmax(minimized_space),
        layers.ArgmaxOutput(dictionary_size),
    ],
    error=crossentropy,
    batch_size=100,
    momentum=0.99,
    nesterov=True,
    verbose=True,
)
embeding_network.train(data, target, epochs=30)

predicted = embeding_network.predict(data)
accuracy = (predicted == target.indices).sum() / len(predicted)
print("Accuracy: {:.2f}%".format(accuracy * 100))

model_storage_file = os.path.join(CURRENT_DIR, 'word-embeding-network.dill')
with open(model_storage_file, 'wb') as f:
    dill.dump(embeding_network, f)
