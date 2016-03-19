import theano
import nltk
from nltk.corpus import stopwords
from sklearn.base import TransformerMixin
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import scipy.sparse as sp


NONE_WORD = ''


class CustomTransformerMixin(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self


class VoidFunctionTransformer(FunctionTransformer):
    def transform(self, X, y=None):
        super(VoidFunctionTransformer, self).transform(X)
        return X


class TokenizeTexts(CustomTransformerMixin):
    def __init__(self, ignore_stopwords=True, *args, **kwargs):
        self.ignore_stopwords = ignore_stopwords
        super(TokenizeTexts, self).__init__(*args, **kwargs)

    def transform(self, X, y=None):
        texts = X
        tokenized_texts = []
        stoplist = []

        if self.ignore_stopwords:
            stoplist = stopwords.words('english')

        for text in texts:
            tokenized_text = []
            for word in nltk.word_tokenize(text.lower()):
                if word not in stoplist:
                    tokenized_text.append(word)

            tokenized_texts.append(tokenized_text)
        return tokenized_texts


class PrepareTrainingData(CustomTransformerMixin):
    def __init__(self, dictionary, window_size=2, *args, **kwargs):
        self.dictionary = dictionary
        self.window_size = window_size
        super(PrepareTrainingData, self).__init__(*args, **kwargs)

    def transform(self, X, y=None):
        texts = X
        dictionary = self.dictionary
        non_word_index = dictionary.index(NONE_WORD)
        window_size = self.window_size
        train_data, target_data = [], []

        for text in texts:
            # Add left padding
            cleaned_text = [non_word_index] * window_size
            for word in text:
                if word in dictionary:
                    cleaned_text.append(dictionary.index(word))

            # Add right padding
            cleaned_text += [non_word_index] * window_size

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
