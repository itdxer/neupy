import os
import re
import pickle
import logging
from collections import defaultdict, namedtuple
from urllib.parse import urljoin, urlparse

import nltk
import numpy as np
import scipy.sparse as sp

from graph import DirectedGraph
from pagerank import pagerank
from htmltools import iter_html_files, ParseHTML


logging.basicConfig(format='[%(levelname)-5s] %(message)s',
                    level=logging.DEBUG)

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


def iter_documents(directory):
    logging.info("Collecting documents from the directory (%s)", directory)
    Document = namedtuple("Document", "filename filepath url links html text")

    for filepath in iter_html_files(directory):
        current_page_url = url_from_file(filepath)
        filename = os.path.basename(filepath)

        if ignore_link(current_page_url):
            logging.debug('Skip "%s", bacause file is defined in the '
                          'ignore list', filename)
            continue

        with open(filepath) as html_file:
            html = html_file.read()
            html = ParseHTML(html, url=current_page_url)
            text = html.text()

            links = []
            for link in url_filter(html.links()):
                links.append(link)

            if text is None:
                logging.debug('Skip "%s", because text is missed',
                              filename)
                continue

            yield Document(filename, filepath, current_page_url,
                           links, html, text)


if __name__ == '__main__':
    logging.info("Started building index")

    documents = []
    vocabulary = {}

    term_frequency = defaultdict(int)
    link_graph = DirectedGraph()

    index_pointers = [0]
    indeces = []
    data = []

    for document in iter_documents(SITE_DIR):
        logging.debug('Processing "%s"', document.filename)

        link_graph.add_node(document.url)
        for link in document.links:
            link_graph.add_edge(document.url, link)

        text = document.text
        text = text.lower().replace('.', ' ').replace('=', ' ')

        for term in nltk.word_tokenize(text):
            if term not in vocabulary:
                vocabulary[term] = len(vocabulary)

            termid = vocabulary[term]
            term_frequency[termid] += 1

            indeces.append(termid)
            data.append(1)

        index_pointers.append(len(indeces))
        documents.append(document._asdict())

    n_documents = len(documents)
    n_terms = len(vocabulary)

    if n_documents == 0:
        raise OSError("Cannot find site documents. Probably site "
                      "hasn't been build yet.")

    logging.info("Found {} HTML files".format(n_documents))
    logging.info("Found {} terms".format(n_terms))

    logging.info("Calculation TF and IDF")
    frequencies = sp.csr_matrix((data, indeces, index_pointers),
                                shape=(n_documents, n_terms))
    df = (frequencies >= 1).sum(axis=0)
    idf = np.log((n_documents / df) + 1)
    idf = np.asarray(idf)

    tf = np.log1p(frequencies)
    tf.data += 1

    logging.info("Applying PageRank")
    rank = pagerank(link_graph)

    logging.info("Saving index")
    save_index([documents, vocabulary, tf, idf, rank])

    logging.info("Index build was finished succesfuly")
