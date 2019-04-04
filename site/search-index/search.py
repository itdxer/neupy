import os
import pickle
import argparse

import nltk
import numpy as np
import scipy.sparse as sp

from build import INDEX_FILE


parser = argparse.ArgumentParser()
parser.add_argument("-q", "--query", help="search query", default=None)
parser.add_argument("-n", "--number-of-results", default=10, type=int,
                    dest="n_results", help="maximum number of top results")


def load_index():
    if not os.path.exists(INDEX_FILE):
        raise IndexError("Cannot find index file. Build a new file with "
                         "command:\n    python search-index/build.py")

    with open(INDEX_FILE, 'rb') as f:
        return pickle.load(f)


def answer_query(query, index, n_results=10):
    documents, vocabulary, tf, idf, pagerank = index
    # document_norm = sp.linalg.norm(tf, axis=1)

    query_tokens = nltk.word_tokenize(query.lower())
    indices = [vocabulary[t] for t in query_tokens if t in vocabulary]

    if not indices:
        print('\nCannot find documents relevant to the specified query')
        return

    document_vector = tf[:, indices]
    query_vector = idf[indices]
    rank = document_vector.dot(query_vector)  # / document_norm

    document_ids, = rank.nonzero()

    similarity_score = rank[document_ids].T
    pagerank_score = pagerank[document_ids]
    rank = similarity_score + pagerank_score
    order = np.asarray(rank.argsort())

    print("Found {} relevant documents".format(len(document_ids)))

    order = order[:n_results]
    for i, index in enumerate(reversed(order), start=1):
        document_id = document_ids[index]
        document = documents[document_id]
        score = rank[index]

        print("")
        print("{}) {}".format(i, document['uri']))
        print("   Total Score: {}".format(score))
        print("   PageRank Score: {}".format(pagerank_score[index]))
        print("   Similarity Score: {}".format(similarity_score[index]))


if __name__ == '__main__':
    print("Search results on NeuPy website")

    index = load_index()
    args = parser.parse_args()

    if args.query is not None:
        answer_query(args.query, index, args.n_results)

    else:
        while True:
            print('\n' + '-' * 60)
            query = input('Query: ')
            answer_query(query, index, args.n_results)
