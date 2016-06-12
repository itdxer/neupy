from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.base import TransformerMixin
from sklearn.preprocessing import FunctionTransformer


class CustomTransformerMixin(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self


class VoidFunctionTransformer(FunctionTransformer):
    def transform(self, X, y=None):
        super(VoidFunctionTransformer, self).transform(X)
        return X


class TokenizeText(CustomTransformerMixin):
    def __init__(self, ignore_stopwords=True, *args, **kwargs):
        self.ignore_stopwords = ignore_stopwords
        super(TokenizeText, self).__init__(*args, **kwargs)

    def transform(self, texts, y=None):
        tokenizer = RegexpTokenizer(r'[a-z]+|\d+')

        tokenized_texts = []
        stoplist = []

        if self.ignore_stopwords:
            stoplist = stopwords.words('english')

        for text in texts:
            tokenized_text = []
            for word in tokenizer.tokenize(text.lower()):
                if word not in stoplist:
                    tokenized_text.append(word.strip())

            tokenized_texts.append(tokenized_text)
        return tokenized_texts


class IgnoreUnknownWords(CustomTransformerMixin):
    def __init__(self, dictionary, *args, **kwargs):
        self.dictionary = dictionary

        if len(dictionary) < 0:
            raise ValueError("Dictionary is empty")

        super(IgnoreUnknownWords, self).__init__(*args, **kwargs)

    def transform(self, texts, y=None):
        dictionary = self.dictionary
        cleaned_texts = []

        for text in texts:
            cleaned_text = [word for word in text if word in dictionary]
            cleaned_texts.append(cleaned_text)

        return cleaned_texts
