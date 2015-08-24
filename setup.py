import os

from setuptools import setup, find_packages

import neuralpy


CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))


def get_requirements():
    try:
        requirements_file = os.path.join(CURRENT_DIR, 'requirements/main.txt')
        with open(requirements_file) as f:
            return f.read().splitlines()
    except IOError:
        # Simple hack for `tox` test.
        return []


setup(
    # Info
    name='neural-python',
    version=neuralpy.__version__,
    description=neuralpy.__doc__,

    # Author
    author='Yurii Shevhcuk',
    author_email='mail@itdxer.com',

    # Package
    packages=find_packages(),
    install_requires=get_requirements(),
    include_package_data=True,
    zip_safe=False,

    # Command
    entry_points={
        'console_scripts': ['neuralpy = neuralpy.commands.main:main'],
    },
)
