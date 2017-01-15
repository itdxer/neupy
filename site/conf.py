# -*- coding: utf-8 -*-

# flake8: noqa

import os
import re
import sys
import importlib

import tinkerer
import tinkerer.paths

# Change this to the name of your blog
project = 'NeuPy'

# Change this to the tagline of your blog
tagline = 'Neural Networks in Python'

# Change this to the description of your blog
description = ('NeuPy is a Python library for Artificial Neural Networks. '
               'NeuPy supports many different types of Neural Networks '
               'from a simple perceptron to deep learning models.')

# Change this to your name
author = 'Yurii Shevchuk'

# Change this to your copyright string
copyright = '2015 - 2017, ' + author

# Change this to your blog root URL (required for RSS feed)
website = 'http://neupy.com'

# **************************************************************
# More tweaks you can do
# **************************************************************

# Add your Disqus shortname to enable comments powered by Disqus
disqus_shortname = "neupy"

# Change your favicon (new favicon goes in _static directory)
html_favicon = '_static/favicon.ico'

# Pick another Tinkerer theme or use your own
html_theme = 'flat'

# Theme-specific options, see docs
html_theme_options = {}

# Link to RSS service like FeedBurner if any, otherwise feed is
# linked directly
rss_service = None

# Generate full posts for RSS feed even when using "read more"
rss_generate_full_posts = False

# Number of blog posts per page
posts_per_page = 10

# Character use to replace non-alphanumeric characters in slug
slug_word_separator = '_'

# Set to page under /pages (eg. "about" for "pages/about.html")
landing_page = 'home'

# Set to override the default name of the first page ('Home')
first_page_title = 'Articles'

# **************************************************************
# Edit lines below to further customize Sphinx build
# **************************************************************

# Add other Sphinx extensions here
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.linkcode',
    'sphinx.ext.mathjax',

    'tinkerer.ext.blog',
    'tinkerer.ext.disqus',

    'numpydoc',
]

autodoc_default_flags = []

# Add other template paths here
templates_path = ['_templates']

# Add other static paths here
html_static_path = ['_static', tinkerer.paths.static]

# Add other theme paths here
html_theme_path = ['_themes', tinkerer.paths.themes]

# Add file patterns to exclude from build
exclude_patterns = ['drafts/*', '_templates/*']

# Add templates to be rendered in sidebar here
html_sidebars = {
    '**': [
        'recent.html',
        'installation.html',
        'searchbox.html',
        'issues.html',
        'old-versions.html',
    ],
}

autodoc_default_flags = ['members', 'undoc-members']

# Add an index to the HTML documents.
html_use_index = False

# **************************************************************
# Autodoc settigs
# **************************************************************

# Add module folder in path to make it visible for autodoc extention
module_path = os.path.abspath(os.path.join('..', 'neupy'))
if module_path not in sys.path:
    sys.path.append(module_path)

autoclass_content = "class"

# **************************************************************
# NumPyDoc
# **************************************************************

numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = True

# **************************************************************
# Customizations
# **************************************************************

GITHUB_REPO = 'https://github.com/itdxer/neupy/tree/master'


def linkcode_resolve(domain, info):
    if domain == 'py' and info['module']:
        filename = info['module'].replace('.', '/')
        return "{}/{}.py".format(GITHUB_REPO, filename)


def get_module_for_class(classname, moduletype):
    """
    Return model for class just using its name and module type.
    """
    available_module_types = {
        'network': 'neupy.algorithms',
        'layer': 'neupy.layers',
        'plot': 'neupy.plots',
    }

    if moduletype not in available_module_types:
        raise ValueError("Invalid module type `{}`".format(moduletype))

    modulepath = available_module_types[moduletype]
    algorithms_module = importlib.import_module(modulepath)
    network_class = getattr(algorithms_module, classname)

    if network_class is None:
        raise ImportError("Can't import network class {}".format(classname))

    return network_class.__module__


def process_docstring(app, what, name, obj, options, lines):
    """
    Function replaces labeled class names to real links in
    the documentation.

    Available types:
    - :network:`NetworkClassName`
    - :layer:`LayerClassName`
    - :plot:`function_name`
    """
    labels = ['network', 'layer', 'plot']
    labels_regexp = '|'.join(labels)

    regexp = re.compile(
        r'\:({})+?\:'
        r'\`(.+?)(|\s<.+?>)\`'.format(labels_regexp)
    )

    if options is not None:
        # Do not show information about parent classes
        options['show-inheritance'] = False

    for i, line in enumerate(lines):
        # Replace values one by one to make sure that there is no overlaps.
        replacement = regexp.search(line)
        while replacement:
            moduletype, display_name, classname = replacement.groups()

            if not classname:
                classname = display_name
            else:
                # Remove first second and last symbols.
                # They must be: \s, < and > respectively.
                classname = classname[2:-1]

            module = get_module_for_class(classname, moduletype)

            if not classname:
                newline_pattern = r':class:`\2 <{}\.\2>`'.format(module)
            else:
                newline_pattern = r':class:`\2 <{}\.{}>`'.format(module,
                                                                 classname)

            line = regexp.sub(newline_pattern, line, count=1)
            line = line.replace('\.', '.')
            lines[i] = line
            replacement = regexp.search(line)


def preprocess_texts(app, docname, source):
    """
    Call the same behaviour for all texts, not only for autodoc.
    """
    process_docstring(app, None, None, None, None, source)


def process_arguments(app, what, name, obj, options, signature,
                      return_annotation):
    """
    Exclude arguments for classes in the documentation.
    """
    if what == 'class':
        return (None, None)


def setup(app):
    """
    Function with reserved name that would be trigger by Sphinx
    if it will find such function in configuration file.
    """
    app.connect('autodoc-process-docstring', process_docstring)
    app.connect('autodoc-process-signature', process_arguments)
    app.connect('source-read', preprocess_texts)

# **************************************************************
# Do not modify below lines as the values are required by
# Tinkerer to play nice with Sphinx
# **************************************************************

source_suffix = tinkerer.source_suffix
master_doc = tinkerer.master_doc
version = tinkerer.__version__
release = tinkerer.__version__

html_title = project
html_show_sourcelink = False
html_add_permalinks = None
