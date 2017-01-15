import os
import re
import pickle
import logging
from collections import defaultdict, namedtuple
from urllib.parse import urljoin, urlparse

import nltk
import numpy as np
import scipy.sparse as sp

from webgraph import WebPageGraph, Link
from pagerank import pagerank
from htmltools import iter_html_files, ParseHTML


logging.basicConfig(format='[%(levelname)-5s] %(message)s',
                    level=logging.DEBUG)

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
INDEX_DIR = os.path.join(CURRENT_DIR, 'index-files')
INDEX_FILE = os.path.join(INDEX_DIR, 'index.pickle')
SITE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'blog', 'html'))
SITE_ROOT = 'http://neupy.com'


def make_url_from_file(filepath):
    _, url = filepath.split(SITE_DIR)
    return urljoin(SITE_ROOT, url)


def ignore_link(link):
    patterns = [
        # Base pages
        '/rss.html',
        '/index.html',
        '/master.html',
        '/search.html',
        '/archive.html',
        '/py-modindex.html',

        # Other pages
        '/pages/home.html',
        '/apidocs/modules.html',

        # Pages that has collected information
        r'/page\d{1,}.html',
        r'.*tags/.+\.html',
        r'.*cheatsheet\.html',

        # Static files
        r'.+(css|js|jpg|png)$',
        r'/_images/.+',
        r'.*\.tar\.gz',
    ]
    uri = urlparse(link)

    for pattern in patterns:
        if re.match(pattern, uri.path):
            return True

        if uri.fragment in ('subpackages', 'submodules'):
            return True

    return False


def url_filter(links):
    filtered_links = []

    for link in links:
        if not ignore_link(link.uri):
            filtered_links.append(link)

    return filtered_links


def save_index(data):
    if not os.path.exists(INDEX_DIR):
        os.mkdir(INDEX_DIR)

    with open(INDEX_FILE, 'wb') as f:
        pickle.dump(data, f)


def collect_documents(directory):
    logging.info("Collecting documents from the directory (%s)", directory)
    Document = namedtuple("Document", "filename filepath uri links html text")

    documents = []
    for filepath in iter_html_files(directory):
        current_page_url = make_url_from_file(filepath)
        filename = os.path.basename(filepath)

        if ignore_link(current_page_url):
            logging.debug('Skip "%s", bacause file is defined in the '
                          'ignore list', filename)
            continue

        html = ParseHTML.fromfile(filepath, current_page_url)
        text = html.text()

        if not text:
            logging.debug('Skip "%s", because text is missed', filename)
            continue

        for subdocument in html.subdocuments():
            if ignore_link(subdocument.uri):
                logging.debug('Skip "%s", because URL is defined in the '
                              'ignore list', subdocument.uri)

            else:
                doc = Document(filename, filepath, subdocument.uri,
                               url_filter(subdocument.links),
                               subdocument.html, subdocument.text)
                documents.append(doc)

    return documents


if __name__ == '__main__':
    logging.info("Started building index")

    documents = []
    vocabulary = {}
    term_frequency = defaultdict(int)

    index_pointers = [0]
    indeces = []
    data = []

    logging.info("Collecting documents")
    all_documents = collect_documents(SITE_DIR)

    logging.info("Define relations between documents")
    webgraph = WebPageGraph.create_from_documents(all_documents)

    for document in all_documents:
        logging.debug('Processing "%s"', document.uri)

        text = document.text
        text = text.lower().replace('.', ' ').replace('=', ' ')

        anchor_texts = []
        for _, link in webgraph.page_linked_by(Link(document.uri)):
            if link.text:
                anchor_texts.append(link.text)

        text = ' '.join([text] + anchor_texts)

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

    logging.info("Found {} documents".format(n_documents))
    logging.info("Found {} terms".format(n_terms))

    logging.info("Calculation TF and IDF")
    frequencies = sp.csr_matrix((data, indeces, index_pointers),
                                shape=(n_documents, n_terms))
    df = (frequencies >= 1).sum(axis=0)
    idf = np.log((n_documents / df) + 1)
    idf = np.asarray(idf)[0]

    tf = np.log1p(frequencies)
    tf.data += 1

    logging.info("Applying PageRank")
    rank = webgraph.pagerank()

    logging.info("Saving index")
    save_index([documents, vocabulary, tf, idf, rank])

    logging.info("Index build was finished succesfuly")
