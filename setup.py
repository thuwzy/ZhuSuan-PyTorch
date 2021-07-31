#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path
import re

here = path.abspath(path.dirname(__file__))

# Get the version string
with open(path.join(here, 'zhusuan', '__init__.py'), encoding='utf-8') as f:
    version = re.search(r'__version__ = \'(.*?)\'', f.read()).group(1)

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    readme = f.read()

setup(
    name='zhusuan-torch',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=version,

    description='ZhuSuan-PyTorch: A Library for Bayesian Deep Learning',
    long_description=readme,

    # The project's main homepage.
    url='https://github.com/thuwzy/ZhuSuan-PyTorch',

    # Author details
    author='ZhuSuan contributors',
    author_email='wang-zy21@mails.tsinghua.edu.cn',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],

    # What does your project relate to?
    keywords='zhusuan generative probabilistic bayesian deep learning paddle',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['tests', 'tests.*',
                                    'examples', 'examples.*']),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['numpy>=1.19.1',],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        'dev': [
            'Sphinx>=4.0.2',
            'sphinx_rtd_theme',
            'sphinxcontrib-bibtex>=2.3.0',
            'pep8',
            'scipy',
            'coverage',
            'mock'
        ],
        'examples': [
            'scipy',
            'matplotlib',
            'scikit-image',
            'progressbar2'
        ],
    },
)