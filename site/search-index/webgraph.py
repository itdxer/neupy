import pprint
from collections import OrderedDict, namedtuple
from urllib.parse import urlparse

import six
import numpy as np
import scipy.sparse as sp

from pagerank import pagerank


class Link(object):
    def __init__(self, uri, text=''):
        self.uri = uri
        self.text = text

    def __eq__(self, link):
        if isinstance(link, six.string_types):
            return self.uri == link
        return self.uri == link.uri

    def __hash__(self):
        return hash(self.uri)

    def __reduce__(self):
        arguments = (self.uri, self.text)
        return (self.__class__, arguments)

    def __repr__(self):
        if not self.text:
            return "<Link uri:{}>".format(self.uri)
        return "<Link uri:{} text:{}>".format(self.uri, self.text)


class WebPageGraph(object):
    def __init__(self):
        self.graph = OrderedDict()

    @classmethod
    def create_from_documents(cls, documents):
        webgraph = cls()

        for document in documents:
            webgraph.add_page(Link(document.uri))

        for document in documents:
            for link in document.links:
                if webgraph.has_page(link):
                    webgraph.connect_pages(Link(document.uri), link)
                    continue

                link_uri = urlparse(link.uri)
                link_url = link_uri.path

                for existed_link, _ in webgraph:
                    existed_link_uri = urlparse(existed_link.uri)
                    existed_link_url = existed_link_uri.path

                    if link_url == existed_link_url:
                        webgraph.connect_pages(
                            Link(document.uri),
                            Link(existed_link.uri, link.text))
        return webgraph


    def add_page(self, page):
        if page not in self.graph:
            self.graph[page] = []

    def has_page(self, page):
        return page in self.graph

    def connect_pages(self, page_1, page_2):
        self.add_page(page_1)
        self.add_page(page_2)

        self.graph[page_1].append(page_2)

    def page_linked_by(self, page):
        for from_page, to_pages in self.graph.items():
            for to_page in to_pages:
                if to_page == page:
                    yield from_page, to_page

    def pagerank(self):
        return pagerank(self.tomatrix())

    def tomatrix(self):
        links = {}

        for node in self.graph:
            links[node] = len(links)

        rows, cols = [], []

        for from_node, to_nodes in self.graph.items():
            from_node_id = links[from_node]

            for to_node in to_nodes:
                to_node_id = links[to_node]

                rows.append(from_node_id)
                cols.append(to_node_id)

        n_links = len(self)
        data = np.ones(len(rows))
        matrix = sp.coo_matrix((data, (rows, cols)),
                               shape=(n_links, n_links))

        return matrix

    @property
    def pages(self):
        return list(self.graph.keys())

    def __iter__(self):
        for from_page, to_pages in self.graph.items():
            yield from_page, to_pages

    def __len__(self):
        return len(self.graph)

    def __repr__(self):
        graph = list(self.graph.items())
        return pprint.pformat(graph)
