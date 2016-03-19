import theano
from sklearn.pipeline import Pipeline
from neupy import algorithms, layers, environment

from utils import TokenizeTexts


environment.reproducible()
theano.config.floatX = 'float32'

classifier_pipeline = Pipeline([
    ('tokenize_texts', TokenizeTexts(ignore_stopwords=False)),
    ('ignore_unknown_words', IgnoreUnknownWords(dictionary)),
    ('word_embedding', WordEmbedding()),
    ('classify', NeuralNetworkClassifier),
])
