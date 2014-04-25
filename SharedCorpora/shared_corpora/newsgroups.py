from abc import ABCMeta
from collections import Sequence, Iterable
from glob import glob
import os

from numpy import ndarray


data_path = '/Users/stinky/Work/data'
newsgroups_corpus_path = os.path.join(data_path, '20_newsgroups')

_corpus_body_cache = None
_corpus_index_cache = None

article_count = 19997
topic_count = {
    'rec.motorcycles': 1000,
    'comp.sys.mac.hardware': 1000,
    'talk.politics.misc': 1000,
    'soc.religion.christian': 997,
    'comp.graphics': 1000,
    'sci.med': 1000,
    'talk.religion.misc': 1000,
    'comp.windows.x': 1000,
    'comp.sys.ibm.pc.hardware': 1000,
    'talk.politics.guns': 1000,
    'alt.atheism': 1000,
    'comp.os.ms-windows.misc': 1000,
    'sci.crypt': 1000, 'sci.space': 1000,
    'misc.forsale': 1000,
    'rec.sport.hockey': 1000,
    'rec.sport.baseball': 1000,
    'sci.electronics': 1000,
    'rec.autos': 1000,
    'talk.politics.mideast': 1000
}

num_topics = 20
topics = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
          'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos',
          'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt',
          'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian',
          'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']


def extract_field(field, data):
    if field == 'body':
        return data[data.find('\n\n'):data.find('\n-- \n')]
    else:
        headers = data[:data.find('\n\n')]
        items = [h for h in headers.split('\n') if h.lower().startswith(field + ':')]
        items = [h[len(field)+1:].strip() for h in items]

        return '\n'.join(items)


def parse_fn(fn):
    with open(fn) as f:
        data = f.read()
        body = data[data.find('\n\n'):data.find('\n-- \n')]

    return body


def parse_article(fn, group, fields):
    article = {'id': int(os.path.basename(fn)),
               'group': group}
    if fields:
        with open(fn) as f:
            data = f.read()

        for field in fields:
            article[field] = extract_field(field, data)
    return article


def parse_articles(corpus_path, fields=None):
    if not fields:
        fields = ['body', 'subject']

    for path in glob(os.path.join(corpus_path, '*')):
        group = os.path.basename(path)

        for fn in glob(os.path.join(path, '*')):
            yield parse_article(fn, group, fields)


def _get_article_index(corpus_path):
    global _corpus_index_cache

    if not _corpus_index_cache:
        _corpus_index_cache = _build_article_index(corpus_path)

    return _corpus_index_cache

def _build_article_index(corpus_path):
    index = [None] * article_count

    for i, art in enumerate(parse_articles(corpus_path, include_body=False)):
        index[i] = (art['id'], art['group'])

    return index


def _initialize_body_cache(corpus_path):
    art_index = _get_article_index(corpus_path)
    body_index = {}

    for art_id, group in art_index:
        body_index["%d_%s" % (art_id, group)] = parse_fn(os.path.join(corpus_path, group, str(art_id)))

    return body_index


def _get_body_index(corpus_path):
    global _corpus_body_cache

    if not _corpus_body_cache:
        _corpus_body_cache = _initialize_body_cache(corpus_path)

    return _corpus_body_cache


# not usable
class NewsgroupsSequence(Sequence):
    __metaclass__ = ABCMeta

    def __init__(self, corpus_path, indices=None, fields=None):
        self.article_index = _get_article_index(corpus_path)
        self.corpus_path = corpus_path
        self.indices = indices
        self.fields = fields

    def __len__(self):
        if type(self.indices) in [ndarray, list]:
            return len(self.indices)
        else:
            return article_count

    def _get_index(self, index):
        raise NotImplementedError

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self._get_index(i) for i in xrange(*index.indices(len(self)))]
        elif isinstance(index, Iterable):
            return [self._get_index(i) for i in index]
        elif isinstance(index, int):
            return self._get_index(index)
        else:
            raise TypeError


class GroupSequence(NewsgroupsSequence):
    def _get_index(self, index):
        return self.article_index[index]['group']


class ArticleSequence(NewsgroupsSequence):
    def __init__(self, corpus_path, indices=None, preprocessor=None, ):
        super(ArticleSequence, self).__init__(corpus_path, indices=indices)

        self.preprocessor = preprocessor

    def _get_index(self, index):
        art_id, group = self.article_index[index]
        body = _get_body_index(self.corpus_path)["%d_%s" % (art_id, group)]

        if self.preprocessor:
            body = self.preprocessor(body)

        return unicode(body, errors='ignore')

