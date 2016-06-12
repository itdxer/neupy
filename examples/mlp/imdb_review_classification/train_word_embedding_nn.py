import os

import pandas as pd
from neupy import environment

from src.word_embedding_nn import WordEmbeddingNN
from src.preprocessing import TokenizeText
from src.utils import create_logger, REVIEWS_FILE, WORD_EMBEDDING_NN


logger = create_logger(__name__)
environment.reproducible()

if not os.path.exists(REVIEWS_FILE):
    raise EnvironmentError("Cannot find reviews.csv file. Probably you "
                           "haven't run `loadata.py` script yet.")

data = pd.read_csv(REVIEWS_FILE, sep='\t')
train_data = data[data.type == 'train']
documents = train_data.text.values

logger.info("Tokenizing train data")
text_tokenizer = TokenizeText(ignore_stopwords=False)
word2vec = WordEmbeddingNN(size=100, workers=4, min_count=5, window=10)

text = text_tokenizer.transform(documents)

logger.info("Building vocabulary")
word2vec.build_vocab(text)
word2vec.train(text, n_epochs=10)

logger.info("Saving model into the {} file".format(WORD_EMBEDDING_NN))
word2vec.save(WORD_EMBEDDING_NN)
