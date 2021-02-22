#!/usr/bin/env python
from setuptools import setup, find_packages

with open("requirements.txt", encoding="utf-8") as req_fp:
    install_requires = req_fp.readlines()

setup(
    name='citrus_nlp',
    version='0.1',
    description='Post-hoc explanations for NLP model predictions/',
    author='Google Research',
    url='https://github.com/google-research/citrus-nlp',
    license='Apache License',
    install_requires=install_requires,
    packages=find_packages(exclude=[]),
    python_requires='>=3.7',
    project_urls={
        'Source': 'https://github.com/google-research/citrus-nlp',
        'Tracker': 'https://github.com/google-research/citrus-nlp/issues',
    },
    entry_points={
        'console_scripts': [
        ],
    }
)
