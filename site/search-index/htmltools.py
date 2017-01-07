import os
from collections import namedtuple
from urllib.parse import urlparse, urljoin

from bs4 import BeautifulSoup


def iter_html_files(directory):
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.html'):
                yield os.path.join(path, filename)


class ParseHTML(object):
    def __init__(self, html, url):
        self.raw_html = html
        self.html = BeautifulSoup(html, "html.parser")
        self.current_page_url = url

    def text(self):
        articles = self.html.select('div.main-container .body article')

        if not articles:
            return ''

        content = articles[0]

        for tag in content.findAll(['script', 'style']):
            tag.extract()

        return content.text

    def extract_links(self, html):
        for anchor in html.findAll('a'):
            uri = urlparse(anchor['href'])

            if uri.netloc in ('neupy.com', ''):
                yield urljoin(self.current_page_url, uri.path)

    def links(self):
        return self.extract_links(self.html)

    def subdocuments(self):
        Subdocument = namedtuple("Subdocument", "url_fragment links html text")

        subdocuments = []
        subdocs = self.html.select('dl.function,dl.class')
        if subdocs:
            for subdoc in subdocs:
                first_child = subdoc.findChild()
                subdocuments.append(Subdocument(
                    url_fragment=first_child['id'],
                    links=list(self.extract_links(subdoc)),
                    html=str(subdoc),
                    text=subdoc.text,
                ))

        else:
            # TODO: Ignore sections with sections inside
            subdocs = self.html.select('div.section')

            for subdoc in subdocs:
                for section in subdoc.select('div.section,div#contents'):
                    section.extract()

                subdocuments.append(Subdocument(
                    url_fragment=subdoc['id'],
                    links=list(self.extract_links(subdoc)),
                    html=str(subdoc),
                    text=subdoc.text,
                ))

        return subdocuments


    def __reduce__(self):
        arguments = (self.raw_html, self.current_page_url)
        return (self.__class__, arguments)
