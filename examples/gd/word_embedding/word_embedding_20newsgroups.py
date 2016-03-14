from collections import Counter
from itertools import chain

import theano
import theano.tensor as T
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.pipeline import Pipeline
from neupy import algorithms, layers, environment
from neupy.utils import asint

from utils import accuracy_score, AverageLinearLayer, crossentropy
from preprocessing import (TokenizeTexts, VoidFunctionTransformer,
                           PrepareTrainingData)


environment.reproducible()
theano.config.floatX = 'float32'


class WordEmbedding(algorithms.Momentum):
    def __init__(self, minimized_space, min_frequency=10, **options):
        self.dictionary = []
        self.n_words = None
        self.min_frequency = min_frequency
        self.minimized_space = minimized_space
        self.default_options = options

    def build_dictionary(self, data):
        if hasattr(self, 'connection'):
            raise ValueError('Neural Network has already created. '
                             'Functionality that helps extend dictionary is '
                             'not implemented.')

        counted_words = Counter(chain(*data))
        most_frequent_words = counted_words.most_common(self.n_words)
        dictionary = self.dictionary

        for word, frequency in most_frequent_words:
            if frequency >= self.min_frequency:
                dictionary.append(word)

        self.n_words = len(dictionary)

        connection = [
            AverageLinearLayer(self.n_words),
            layers.Softmax(self.minimized_space),
            layers.ArgmaxOutput(self.n_words),
        ]
        super(WordEmbedding, self).__init__(connection, **self.default_options)

    def init_methods(self):
        super(WordEmbedding, self).init_methods()

        network_input = self.variables.network_input
        self.methods.transform_to_vector = theano.function(
            [network_input],
            self.input_layer.output(network_input)
        )

        words_id_list = self.variables.word_id = T.ivector()
        weight = self.input_layer.weight
        words_matrix = weight[words_id_list, :]

        all_word_vectors_length = weight.norm(L=2, axis=1).reshape((-1, 1))
        word_vectors_length = words_matrix.norm(L=2, axis=1).reshape((-1, 1))

        unit_words_matrix = words_matrix / word_vectors_length
        unit_weight = weight / all_word_vectors_length
        cosine_similarities = unit_words_matrix.dot(unit_weight.T)

        n_words = words_id_list.shape[0]
        the_same_words = cosine_similarities[T.arange(n_words), words_id_list]
        # Similarity within the same words will be equal to 1.
        # That's why we need make them equal to 0
        cosine_similarities = T.set_subtensor(the_same_words, 0)

        self.methods.cosine_similarties = theano.function(
            [words_id_list], cosine_similarities
        )

    def transform_to_vector(self, text):
        return self.methods.transform_to_vector(text)

    def most_similar(self, words, n_similar=10):
        identifiers = [self.dictionary.index(word) for word in words]
        identifiers = asint(identifiers)
        similarities = self.methods.cosine_similarties(identifiers)

        similarities_indeces = similarities.argsort(axis=1)
        similarities.sort(axis=1)

        n_most_similar_vales = similarities[:, -n_similar:][:, ::-1]
        n_most_similar_indeces = similarities_indeces[:, -n_similar:][:, ::-1]

        dictionary = np.array(self.dictionary)
        n_most_similar_words = dictionary[n_most_similar_indeces]

        return n_most_similar_words, n_most_similar_vales


data = pd.read_csv('amazon_cells_labelled.txt', sep='\t',
                   header=None, names=['review', 'sentiment'])
reviews = data.review.values

embedding_network = WordEmbedding(
    minimized_space=400,
    min_frequency=5,

    error=crossentropy,
    batch_size=100,
    momentum=0.99,
    nesterov=True,
    verbose=True,
    shuffle_data=True,
    step=0.1,
)

text_preprocessing = Pipeline([
    ('tokenize_texts', TokenizeTexts(ignore_stopwords=False)),
    ('build_dictionary', VoidFunctionTransformer(
        func=embedding_network.build_dictionary,
        validate=False,
    )),
    ('prepare_training_data', PrepareTrainingData(
        dictionary=embedding_network.dictionary,
        window_size=3,
    )),
])
x_train, y_train = text_preprocessing.transform(reviews)
embedding_network.train(x_train, y_train, epochs=30)

# Check the accuracy
predicted = embedding_network.predict(x_train)
accuracy = accuracy_score(y_train.indices, predicted)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# # Visualize embedded words
# word_vectors = embedding_network.input_layer.weight.get_value()
# tsne = manifold.LocallyLinearEmbedding(n_components=2)
# minimized_word_vectors = tsne.fit_transform(word_vectors)
# id2token = {v:k for k, v in embedding_network.dictionary.items()}
#
# fig, ax = plt.subplots(1, 1)
# ax.set_ylim(-15, 15)
# ax.set_xlim(-15, 15)
# for i, row in enumerate(minimized_word_vectors[:500]):
#     ax.annotate(id2token[i], row)
#
# plt.show()
