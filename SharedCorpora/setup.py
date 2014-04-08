from setuptools import setup

# noinspection PyUnresolvedReferences
import multiprocessing

setup(
    name='SharedCorpora',
    version='0.1',
    packages=['shared_corpora'],
    url='',
    license='',
    author='',
    author_email='',
    description='',
    test_suite='nose.collector', requires=['gensim', 'psutil', 'lxml']
)
