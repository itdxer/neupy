import os
import pickle

import nltk
import numpy as np
import scipy.sparse as sp

from build import INDEX_FILE


def normalize(vector):
    if sp.issparse(vector):
        vector_norm = sp.linalg.norm(vector, axis=1)
    else:
        vector_norm = np.linalg.norm(vector, axis=1)

    vector_norm[vector_norm == 0] = 1
    vector_norm = np.expand_dims(vector_norm, axis=1)

    return vector / vector_norm


def cosine_similarity(doc_vector, query_vector):
    return normalize(doc_vector).dot(normalize(query_vector).T)


def similarity(doc_vector, query_vector):
    if query_vector.size == 1:
        score = doc_vector * query_vector
        return score / score.max()
    return cosine_similarity(doc_vector, query_vector)


def load_index():
    if not os.path.exists(INDEX_FILE):
        raise IndexError("Cannot find index file. Build a new file with "
                         "command:\n    python search-index/build.py")

    with open(INDEX_FILE, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    print("Search on NeuPy website")
    documents, vocabulary, tf, idf, pagerank = load_index()

    while True:
        print('\n' + '-' * 60)
        # query = input('Query: ')

        query_tokens = nltk.word_tokenize(query.lower())
        indeces = [vocabulary[t] for t in query_tokens if t in vocabulary]

        if not indeces:
            print('\nYour search did not match any documents')
            break

        rank = similarity(tf[:, indeces], idf[:, indeces])
        document_ids, _ = rank.nonzero()

        similarity_score = rank[document_ids].T
        pagerank_score = pagerank[document_ids]
        rank = similarity_score + 2 * pagerank_score
        # rank = np.expand_dims(pagerank[document_ids], axis=0)
        order = np.asarray(rank.argsort())[0][::-1]

        print("Found {} results".format(len(document_ids)))
        print('')

        for i, index in enumerate(order, start=1):
            document_id = document_ids[index]
            document = documents[document_id]
            score = rank[0, index]

            print("{}) {}".format(i, document['url']))
            print("   Total Score: {}".format(score))
            print("   PageRank Score: {}".format(pagerank_score[index]))
            print("   Similarity Score: {}".format(similarity_score[0, index]))
            print("")

            if i == 5:
                break

        break
