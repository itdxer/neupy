import os
from collections import namedtuple
from urllib.parse import urlparse, urljoin

from bs4 import BeautifulSoup

from webgraph import Link


def iter_html_files(directory):
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.html'):
                yield os.path.join(path, filename)


def extract_title(html):
    headers = html.select("h1,h2,h3,h4,h5")
    first_header = headers[0]
    return first_header.text


class ParseHTML(object):
    def __init__(self, html, url):
        self.raw_html = html
        self.html = BeautifulSoup(html, "html.parser")
        self.url = url

    @classmethod
    def fromfile(cls, filepath, url):
        with open(filepath) as html_file:
            html = html_file.read()
            return cls(html, url)

    def text(self):
        articles = self.html.select('div.main-container .body article')

        if not articles:
            return ''

        content = articles[0]

        for tag in content.findAll(['script', 'style', 'noscript']):
            tag.extract()

        return content.text

    def extract_links(self, html):
        for anchor in html.findAll('a'):
            uri = urlparse(anchor['href'])

            if uri.netloc in ('neupy.com', ''):
                url = urljoin(self.url, uri.path)
                if uri.fragment:
                    url = url + "#" + uri.fragment

                yield Link(uri=url, text=anchor.text)

    def links(self):
        return self.extract_links(self.html)

    def subdocuments(self):
        Subdocument = namedtuple("Subdocument", "uri links html text title")

        subdocuments = []
        apidocs = self.html.select('dl.function,dl.class')

        if apidocs:
            for subdoc in apidocs:
                first_child = subdoc.findChild()
                object_name = first_child.select('.descname')
                title = object_name[0].text

                subdocuments.append(
                    Subdocument(
                        uri=self.url + "#" + first_child['id'],
                        links=list(self.extract_links(subdoc)),
                        html=str(subdoc),
                        text=subdoc.text,
                        title=title))

        else:
            suptitle = extract_title(self.html.select('.main-container')[0])
            for subdoc in self.html.select('div.section'):
                for section in subdoc.select('div.section,div#contents'):
                    section.extract()

                title = extract_title(subdoc)

                if suptitle and suptitle != title:
                    title = suptitle + " / " + title

                subdocuments.append(
                    Subdocument(
                        uri=self.url + "#" + subdoc['id'],
                        links=list(self.extract_links(subdoc)),
                        html=str(subdoc),
                        text=subdoc.text,
                        title=title))

        if not subdocuments or (len(subdocuments) == 1 and not apidocs):
            subdocumets = [
                Subdocument(
                    uri=self.url,
                    links=self.links(),
                    html=self.raw_html,
                    text=self.text(),
                    title=extract_title(subdoc))]

        return subdocuments

    def __reduce__(self):
        arguments = (self.raw_html, self.url)
        return (self.__class__, arguments)
