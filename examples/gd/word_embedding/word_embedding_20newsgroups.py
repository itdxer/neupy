import os
import string
from collections import Counter
from itertools import chain

import theano
import theano.sparse
import theano.tensor as T
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn import metrics, manifold
from sklearn.datasets import fetch_20newsgroups
from neupy import algorithms, layers
from neupy.core.properties import IntProperty


np.random.seed(0)
theano.config.floatX = 'float32'


class AverageLinearLayer(layers.Layer):
    def output(self, input_value):
        input_value = T.cast(input_value, 'int32')
        summator = self.weight[input_value]
        self.parameters = [self.weight]
        return summator.mean(axis=1)


def crossentropy(expected, predicted, epsilon=1e-10):
    predicted = T.clip(predicted, epsilon, 1.0 - epsilon)
    n_samples = expected.shape[0]

    error = theano.sparse.sp_sum(-expected * T.log(predicted))
    return error / n_samples


def accuracy_score(expected, predicted):
    return (predicted == expected).sum() / len(predicted)


def tokenize_texts(texts):
    stoplist = stopwords.words('english')
    stemmer = PorterStemmer()
    punctuation = string.punctuation

    tokenized_texts = []
    for text in texts:
        tokenized_text = []
        for word in nltk.word_tokenize(text.lower()):
            word = ''.join([l for l in word if l not in punctuation])

            if word not in stoplist and len(word) > 1:
                word = stemmer.stem(word)
                tokenized_text.append(word)

        tokenized_texts.append(tokenized_text)
    return tokenized_texts


def prepare_training_data(texts, dictionary, window_size=5):
    train_data, target_data = [], []

    for text in texts:
        cleaned_text = []
        for word in text:
            if word in dictionary:
                cleaned_text.append(dictionary[word])

        for i in range(window_size, len(cleaned_text) - window_size):
            target_word = cleaned_text[i]
            input_context = (
                cleaned_text[i - window_size:i] +
                cleaned_text[i + 1:i + window_size + 1]
            )

            target_data.append(target_word)
            train_data.append(input_context)

    n_samples = len(target_data)
    dictionary_size = len(dictionary)
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


class WordEmbedding(algorithms.Momentum):
    n_words = IntProperty(minval=1, required=True)
    minimized_space = IntProperty(minval=1, required=True)

    def __init__(self, n_words, minimized_space, **options):
        connection = [
            AverageLinearLayer(n_words),
            layers.Softmax(minimized_space),
            layers.ArgmaxOutput(n_words),
        ]

        self.dictionary = {}
        self.n_words = n_words
        self.minimized_space = minimized_space

        super(WordEmbedding, self).__init__(connection, **options)

    def build_dictionary(self, data):
        dictionary = Counter(chain(*data))
        most_frequent_words = dictionary.most_common(self.n_words)

        self.dictionary = {}
        for i, (word, _) in enumerate(most_frequent_words):
            self.dictionary[word] = i

    def init_methods(self):
        super(WordEmbedding, self).init_methods()

        network_input = self.variables.network_input
        self.methods.transform_to_vector = theano.function(
            [network_input],
            self.input_layer.output(network_input)
        )

    def transform_to_vector(self, text):
        return self.methods.transform_to_vector(text)


# Make it possible to use sparse matrix for trainig target
crossentropy.expected_dtype = theano.sparse.csr_matrix
train_newsgroups = fetch_20newsgroups(
    subset='all',
    categories=['rec.autos', 'rec.motorcycles']
)

embedding_network = WordEmbedding(
    n_words=20000,
    minimized_space=100,

    error=crossentropy,
    batch_size=100,
    momentum=0.99,
    nesterov=True,
    verbose=True,
    step=0.1,

    # Decrease step after each epoch
    epochs_step_minimizator=25,
    addons=[algorithms.SimpleStepMinimization]
)

data = tokenize_texts(train_newsgroups.data)
embedding_network.build_dictionary(data)
x_train, y_train = prepare_training_data(
    data,
    embedding_network.dictionary,
    window_size=2
)
embedding_network.train(x_train, y_train, epochs=30)

# Check the accuracy
predicted = embedding_network.predict(x_train)
accuracy = accuracy_score(y_train.indices, predicted)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Visualize embedded words
tsne = manifold.TSNE(n_components=2)
word_vectors = embedding_network.input_layer.weight.get_value()
minimized_word_vectors = tsne.fit_transform(word_vectors)
id2token = {v:k for k, v in embedding_network.dictionary.items()}

fig, ax = plt.subplots(1, 1)
for i, row in enumerate(minimized_word_vectors):
    ax.annotate(id2token[i], row)

plt.show()
