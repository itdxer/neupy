import os
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

    def links(self):
        for anchor in self.html.findAll('a'):
            uri = urlparse(anchor['href'])

            if uri.netloc in ('neupy.com', ''):
                yield urljoin(self.current_page_url, uri.path)

    def subdocuments(self):
        pass

    def __reduce__(self):
        arguments = (self.raw_html, self.current_page_url)
        return (self.__class__, arguments)
