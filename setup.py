import os
import warnings

from setuptools import setup, find_packages

import neupy


CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))


def get_requirements():
    try:
        requirements_file = os.path.join(CURRENT_DIR, 'requirements',
                                         'main.txt')

        with open(requirements_file) as f:
            requirements = f.read()
            return requirements.splitlines()

    except IOError as e:
        warnings.warn("error", e)
        # Simple hack for `tox` test.
        return []


setup(
    # Info
    name='neupy',
    version=neupy.__version__,
    license='MIT',
    url='http://neupy.com',
    description=neupy.__doc__,

    # Author
    author='Yurii Shevhcuk',
    author_email='mail@itdxer.com',

    # Package
    packages=find_packages(),
    install_requires=get_requirements(),
    include_package_data=True,
    zip_safe=False,
    keywords=['neural networks', 'deep learning'],

    # Other
    classifiers=[
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
