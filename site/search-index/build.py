import os
import re
import pickle
import logging
from collections import defaultdict
from urllib.parse import urljoin, urlparse

import nltk
import numpy as np
import scipy.sparse as sp

from graph import DirectedGraph
from pagerank import pagerank
from htmltools import iter_html_files, ParseHTML


logging.basicConfig(level=logging.DEBUG)

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
INDEX_DIR = os.path.join(CURRENT_DIR, 'index-files')
INDEX_FILE = os.path.join(INDEX_DIR, 'index.pickle')
SITE_DIR = os.path.join(CURRENT_DIR, '..', 'blog', 'html')
SITE_ROOT = 'http://neupy.com'


def url_from_file(filepath):
    _, url = filepath.split(SITE_DIR)
    return urljoin(SITE_ROOT, url)


def ignore_link(link):
    patterns = [
        '/pages/versions.html',
        '/index.html',
        '/page\d{1,}.html',
        r'.*cheatsheet.html',
        r'.+(css|js|jpg|png)$',
        r'/_images/.+',
    ]
    uri = urlparse(link)

    for pattern in patterns:
        if re.match(pattern, uri.path):
            return True

    return False


def url_filter(links):
    for link in links:
        if not ignore_link(link):
            yield link


def save_index(data):
    if not os.path.exists(INDEX_DIR):
        os.mkdir(INDEX_DIR)

    with open(INDEX_FILE, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    documents = {}
    vocabulary = {}

    term_frequency = defaultdict(int)
    link_graph = DirectedGraph()

    indeces = []
    data = []
    index_pointers = [0]
    docid = 0

    for html_filepath in iter_html_files(SITE_DIR):
        current_page_url = url_from_file(html_filepath)
        html_filename = os.path.basename(html_filepath)

        if ignore_link(current_page_url):
            continue

        link_graph.add_node(current_page_url)
        logging.debug('Processing "{}"'.format(html_filename))

        with open(html_filepath) as html_file:
            html = html_file.read()
            html = ParseHTML(html, url=current_page_url)
            text = html.text().lower()

            for link in url_filter(html.links()):
                link_graph.add_edge(current_page_url, link)

            if text is None:
                continue

            text = text.replace('.', ' ').replace('=', ' ')

            for term in nltk.word_tokenize(text):
                if term not in vocabulary:
                    vocabulary[term] = len(vocabulary)

                termid = vocabulary[term]
                term_frequency[termid] += 1

                indeces.append(termid)
                data.append(1)

            index_pointers.append(len(indeces))

            documents[docid] = {
                'url': current_page_url,
                'filepath': html_filepath,
                'filename': html_filename,
                'text': text,
            }
            docid += 1

    n_documents = len(documents)
    n_terms = len(vocabulary)

    logging.info("Found {} HTML files".format(n_documents))
    logging.info("Found {} terms".format(n_terms))

    frequencies = sp.csr_matrix((data, indeces, index_pointers),
                                shape=(n_documents, n_terms))
    df = (frequencies >= 1).sum(axis=0)
    idf = np.log((n_documents / df) + 1)
    idf = np.asarray(idf)
    # idf = sp.spdiags(idf, diags=0, m=n_terms, n=n_terms)

    tf = np.log1p(frequencies)
    tf.data += 1

    rank = pagerank(link_graph)
    save_index([documents, vocabulary, tf, idf, rank])
