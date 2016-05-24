import os

import dill
import theano
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn import metrics
from neupy import algorithms, layers, environment

from src.word_embedding_nn import WordEmbeddingNN
from src.preprocessing import TokenizeText, IgnoreUnknownWords
from src.utils import (WORD_EMBEDDING_NN, NN_CLASSIFIER_MODEL, REVIEWS_FILE,
                       create_logger)


logger = create_logger(__name__)

environment.reproducible()
theano.config.floatX = 'float32'

if not os.path.exists(WORD_EMBEDDING_NN):
    raise EnvironmentError("Can't find NN model. File {} doesn't exist {}."
                           "Probably you haven't train it yet. "
                           "Run `train_word_embedding_nn.py` script.")

logger.info("Reading data")
data = pd.read_csv(REVIEWS_FILE, sep='\t')

logger.info("Loading word embedding NN")
word2vec = WordEmbeddingNN.load(WORD_EMBEDDING_NN)

prepare_data_pipeline = Pipeline([
    ('tokenize_texts', TokenizeText(ignore_stopwords=False)),
    ('ignore_unknown_words', IgnoreUnknownWords(dictionary=word2vec.vocab)),
    ('word_embedding', word2vec),
])

classifier = algorithms.RPROP(
    [
        layers.Input(100),
        layers.Relu(200),
        layers.Relu(50),
        layers.Sigmoid(1),
    ],
    error='binary_crossentropy',
    verbose=True,
    shuffle_data=True,

    maxstep=1,
    minstep=1e-7,
)

logger.info("Preparing train data")
train_data = data[data.type == 'train']
texts = train_data.text.values
x_train = prepare_data_pipeline.transform(texts)
y_train = (train_data.sentiment.values == 'pos')

logger.info("Preparing test data")
test_data = data[data.type == 'test']
texts = test_data.text.values
x_test = prepare_data_pipeline.transform(texts)
y_test = (test_data.sentiment.values == 'pos')

classifier.train(x_train, y_train, x_test, y_test, epochs=100)

y_train_predicted = classifier.predict(x_train).round()
y_test_predicted = classifier.predict(x_test).round()

print(metrics.classification_report(y_train_predicted, y_train))
print(metrics.confusion_matrix(y_train_predicted, y_train))
print()
print(metrics.classification_report(y_test_predicted, y_test))
print(metrics.confusion_matrix(y_test_predicted, y_test))

with open(NN_CLASSIFIER_MODEL, 'wb') as f:
    dill.dump(classifier, file=f)
