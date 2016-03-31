import logging
from collections import Counter
from itertools import chain

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from gensim import models
from neupy import environment

from preprocessing import TokenizeTexts


logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)

environment.reproducible()

data = pd.read_csv('amazon_cells_labelled.txt', sep='\t',
                   header=None, names=['review', 'sentiment'])
documents = data.review.values

class WordEmbedding(models.Word2Vec):
    def train(self, data, n_epochs=100):
        train_method = super(WordEmbedding, self).train

        for epoch in range(n_epochs):
            print("Epoch #{}".format(epoch))
            # self.
            train_method(data)

    def transform(self, documents):
        document_vectors = []
        for document in documents:
            document_vector = self[document].sum(axis=0)
            document_vectors.append(document_vectors)
        return np.array(document_vectors)


text_tokenizer = TokenizeTexts(ignore_stopwords=False)
word2vec = WordEmbedding(size=100, workers=4, min_count=1, window=2, hs=1, sg=1)

text = text_tokenizer.transform(documents)
word2vec.build_vocab(text)
word2vec.train(text, n_epochs=20)

word_vectors = []
words = word2vec.vocab.keys()
for word in words:
    word_vectors.append(word2vec[word])

word_vectors = np.array(word_vectors)

# Visualize embedded words
tsne = manifold.TSNE(n_components=2)
minimized_word_vectors = tsne.fit_transform(word_vectors)

fig, ax = plt.subplots(1, 1)
ax.set_ylim(-15, 15)
ax.set_xlim(-15, 15)
for word, vector in zip(words, minimized_word_vectors):
    ax.annotate(word, vector)

plt.show()
