# coding: utf-8

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LISENCE') as f:
    license = f.read()

setup(
    name='xai',
    version='0.2.1',
    description="xikasan's ai library",
    long_description=readme,
    author='xikasan',
    # author_email='',
    url='https://github.com/xikasan/xai',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
