import os
import re
import logging
from collections import defaultdict
from urllib.parse import urljoin, urlparse

import nltk
import numpy as np
import scipy.sparse as sp

from graph import DirectedGraph
from htmltools import iter_html_files, ParseHTML


logging.basicConfig(level=logging.DEBUG)

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
SITE_DIR = os.path.join(CURRENT_DIR, '..', 'blog', 'html')
SITE_ROOT = 'http://neupy.com'


def url_from_file(filepath):
    _, url = filepath.split(SITE_DIR)
    return urljoin(SITE_ROOT, url)


def ignore_link(link):
    patterns = [
        '/pages/versions.html',
        '/index.html',
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


class FeatureExtraction(object):
    def __init__(self, documents):
        pass


if __name__ == '__main__':
    documents = defaultdict()

    vocabulary = defaultdict()
    vocabulary.default_factory = vocabulary.__len__

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

        logging.debug('Processing "{}"'.format(html_filename))

        with open(html_filepath) as html_file:
            html = html_file.read()
            html = ParseHTML(html, url=current_page_url)
            text = html.text()

            for link in url_filter(html.links()):
                link_graph.add_edge(current_page_url, link)

            if text is None:
                continue

            for term in nltk.word_tokenize(text):
                termid = vocabulary[term]
                term_frequency[termid] += 1

                indeces.append(termid)
                data.append(1)

            index_pointers.append(len(indeces))

            documents[docid] = {
                'filepath': html_filepath,
                'filename': html_filename,
                'html': html,
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
    idf = np.log((n_documents + 1) / (df + 1))
    idf = sp.spdiags(idf, diags=0, m=n_terms, n=n_terms)

    tf = np.log1p(frequencies)
    tf.data += 1
