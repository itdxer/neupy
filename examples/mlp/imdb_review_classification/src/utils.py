import os
import logging


CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

REVIEWS_FILE = os.path.join(DATA_DIR, 'reviews.csv')

MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
WORD_EMBEDDING_NN = os.path.join(MODELS_DIR, 'word_embedding.model')
NN_CLASSIFIER_MODEL = os.path.join(MODELS_DIR, 'nn_classifier.model')


def create_logger(name, level=logging.INFO):
    formatter = logging.Formatter(
        fmt='%(asctime)-15s : %(levelname)s : %(message)s'
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger
