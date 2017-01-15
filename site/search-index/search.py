import os
import pickle

import nltk
import numpy as np
import scipy.sparse as sp

from build import INDEX_FILE


def load_index():
    if not os.path.exists(INDEX_FILE):
        raise IndexError("Cannot find index file. Build a new file with "
                         "command:\n    python search-index/build.py")

    with open(INDEX_FILE, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    print("Search on NeuPy website")
    documents, vocabulary, tf, idf, pagerank = load_index()
    # document_norm = sp.linalg.norm(tf, axis=1)

    while True:
        print('\n' + '-' * 60)

        query = input('Query: ')

        query_tokens = nltk.word_tokenize(query.lower())
        indeces = [vocabulary[t] for t in query_tokens if t in vocabulary]

        if not indeces:
            print('\nYour search did not match any documents')
            break

        document_vector = tf[:, indeces]
        query_vector = idf[indeces]
        rank = document_vector.dot(query_vector)  # / document_norm

        document_ids, = rank.nonzero()

        similarity_score = rank[document_ids].T
        pagerank_score = pagerank[document_ids]
        rank = similarity_score + pagerank_score
        order = np.asarray(rank.argsort())

        print("Found {} relevant documents".format(len(document_ids)))

        for i, index in enumerate(reversed(order), start=1):
            document_id = document_ids[index]
            document = documents[document_id]
            score = rank[index]

            print("")
            print("{}) {}".format(i, document['uri']))
            print("   Total Score: {}".format(score))
            print("   PageRank Score: {}".format(pagerank_score[index]))
            print("   Similarity Score: {}".format(similarity_score[index]))
