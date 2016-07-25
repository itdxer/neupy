import numpy as np
from gensim import models

from .utils import create_logger


logger = create_logger(__name__)


class WordEmbeddingNN(models.Word2Vec):
    def train(self, data, n_epochs=100):
        train_method = super(WordEmbeddingNN, self).train
        logger.info("Start training network over {} epochs".format(n_epochs))

        for epoch in range(n_epochs):
            logger.info("Training epoch {} / {}".format(epoch + 1, n_epochs))
            train_method(data)

    def fit(self, X, y, n_epochs=100):
        """
        Function overrides `train` method. This trick adds
        scikit-learn compatibility.
        """
        self.train(X, n_epochs)
        return self

    def fit_transform(self):
        pass

    def transform(self, documents):
        document_vectors = []
        for document in documents:
            document_vector = self[document].mean(axis=0)
            document_vectors.append(document_vector)
        return np.array(document_vectors)
