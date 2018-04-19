#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from pip.req import parse_requirements

install_reqs = parse_requirements('requirements.txt', session=False)
reqs = [str(ir.req) for ir in install_reqs]
dep_links = [str(req_line.url) for req_line in install_reqs]


setup(
    name='bb_trajectory',
    version='0.1',
    description='BeesBook trajectory analysis',
    author='David Dormagen',
    author_email='david.dormagen@fu-berlin.de',
    url='https://github.com/BioroboticsLab/bb_trajectory/',
    install_requires=reqs,
    dependency_links=dep_links,
    packages=['bb_trajectory'],
    package_dir={'bb_trajectory': 'bb_trajectory/'}
)
