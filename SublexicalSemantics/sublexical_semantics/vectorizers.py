from collections import defaultdict
import logging
from operator import itemgetter
import os

from gensim.models import LsiModel, TfidfModel, Word2Vec
from numpy import zeros, load
from scipy import sparse
from scipy.sparse import lil_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import VectorizerMixin


class MultiVectorizer(BaseEstimator):
    def __init__(self, vectorizers=None):
        if not vectorizers:
            raise ValueError

        self.vectorizers = vectorizers

    def fit_transform(self, raw_documents, y=None):
        x = [vect.fit_transform(raw_documents, y) for vect in self.vectorizers]

        x = sparse.hstack(x)

        return x

    def fit(self, raw_documents, y=None):
        for vect in self.vectorizers:
            vect.fit(raw_documents, y)

        return self

    def transform(self, raw_documents):
        x = [vect.transform(raw_documents) for vect in self.vectorizers]

        x = sparse.hstack(x)

        return x


# Simple class to hold Brown cluster assignments read from a model file created by wcluster.
class ClusterIndex(object):
    def __init__(self, file, prefix_size=16):
        self.prefix_size = prefix_size
        self.word_to_cluster, self.cluster_to_word, self.freqs = _parse_cluster_file(file, self.prefix_size)

        self.n_cluster = len(self.cluster_to_word.keys())

    def cluster(self, word):
        return self.word_to_cluster[word]

    def cluster_terms(self, cluster, num_terms=10):
        terms = self.cluster_to_word[cluster]
        freqs = [self.freqs[term] for term in terms]
        sorted_terms = sorted(zip(terms, freqs), key=itemgetter(1), reverse=True)

        return [term[0] for term in sorted_terms[0:num_terms]]

    def clusters(self):
        return self.cluster_to_word.keys()


def _parse_cluster_file(f, prefix_size=16):
    line_num = 0
    word_to_cluster = defaultdict(lambda: None)
    cluster_to_word = defaultdict(lambda: [])
    freqs = defaultdict(lambda: 0)

    for line in f.readlines():
        line_num += 1

        tokens = line.strip().split("\t")

        if len(tokens) != 3:
            logging.warn("Couldn't parse line %d" % line_num)
        else:
            c = tokens[0].strip()[0:prefix_size]
            word = tokens[1].strip()
            freq = int(tokens[2])

            if word_to_cluster.has_key(word) or freqs.has_key(word):
                logging.warn("Duplicate entry \"%s\"" % line)

            word_to_cluster[word] = c
            cluster_to_word[c] += [word]
            freqs[word] = freq

    return word_to_cluster, cluster_to_word, freqs


def _build_vocabulary(cluster_index):
    clusters = sorted(cluster_index.clusters())
    vocab = {}

    for i, c in enumerate(clusters):
        vocab[c] = i

    return vocab


# Transforms input into document vectors based on Brown clusters.
# Based on CountVectorizer class in SKLearn.
class BrownClusterVectorizer(BaseEstimator, VectorizerMixin, TransformerMixin):
    def __init__(self, cluster_fn, prefix_size=16, preprocessor=None):
        self.cluster_fn = cluster_fn

        self.cluster_index = None
        self.vocabulary_ = None

        self.analyzer = 'word'
        self.preprocessor = preprocessor
        self.strip_accents = 'unicode'
        self.lowercase = True
        self.stop_words = None
        self.tokenizer = None
        self.token_pattern = r"(?u)\b\w\w+\b"
        self.input = None
        self.ngram_range = (1, 1)
        self.encoding = 'utf-8'
        self.decode_error = 'ignore'
        self.prefix_size = prefix_size

        self.analyzer_func = None

    # no data needed, this just reads the model
    def fit(self, raw_documents, y=None):
        self.analyzer_func = self.build_analyzer()

        with open(self.cluster_fn) as f:
            self.cluster_index = ClusterIndex(f, prefix_size=self.prefix_size)

        self.vocabulary_ = _build_vocabulary(self.cluster_index)

        return self

    def transform(self, raw_documents):
        x = lil_matrix((len(raw_documents), self.cluster_index.n_cluster))

        for row, doc in enumerate(raw_documents):
            for token in self.analyzer_func(doc):
                c = self.cluster_index.cluster(token)

                if c:
                    x[row, self.vocabulary_[c]] += 1

        return x.tocsr()


class LsiVectorizer(BaseEstimator, VectorizerMixin, TransformerMixin):
    def __init__(self, model_fn, preprocessor=None):
        self.model_fn = model_fn

        self.model = None
        self.tfidf = None

        self.analyzer = 'word'
        self.preprocessor = preprocessor
        self.strip_accents = 'unicode'
        self.lowercase = True
        self.stop_words = None
        self.tokenizer = None
        self.token_pattern = r"(?u)\b\w\w+\b"
        self.input = None
        self.ngram_range = (1, 1)
        self.encoding = 'utf-8'
        self.decode_error = 'ignore'

        self.analyzer_func = None

    def fit(self, raw_documents, y=None):
        self.analyzer_func = self.build_analyzer()

        self.model = LsiModel.load(self.model_fn)

        if os.path.exists(self.model_fn + '.tfidf'):
            self.tfidf = TfidfModel.load(self.model_fn + '.tfidf')

        return self

    def transform(self, raw_documents):
        x = zeros((len(raw_documents), self.model.num_topics))

        for row, doc in enumerate(raw_documents):
            doc = self.model.id2word.doc2bow(self.analyzer_func(doc))

            if self.tfidf:
                topics = self.model[self.tfidf[doc]]
            else:
                topics = self.model[doc]

            for idx, val in topics:
                x[row, idx] = val

        return x


# Creates document vectors from text based on sublexical Word2Vec log-bilinear models
# trained with Gensim. Creates average of term representations in the document.
# Based on CountVectorizer class in SKLearn.
class Word2VecVectorizer(BaseEstimator, VectorizerMixin, TransformerMixin):
    def __init__(self, model_fn, preprocessor=None):
        self.model_fn = model_fn

        self.model = None

        self.analyzer = 'word'
        self.preprocessor = preprocessor
        self.strip_accents = 'unicode'
        self.lowercase = True
        self.stop_words = None
        self.tokenizer = None
        self.token_pattern = r"(?u)\b\w\w+\b"
        self.input = None
        self.ngram_range = (1, 1)
        self.encoding = 'utf-8'
        self.decode_error = 'ignore'

        self.analyzer_func = None

    # no data needed, this just reads the model
    def fit(self, raw_documents, y=None):
        self.analyzer_func = self.build_analyzer()

        self.model = Word2Vec.load(self.model_fn)

        # pick up external data vectors
        if not hasattr(self.model, 'syn0'):
            self.model.syn0 = load(self.model_fn + '.syn0.npy')

        if not hasattr(self.model, 'syn1'):
            self.model.syn0 = load(self.model_fn + '.syn1.npy')

        return self

    def transform(self, raw_documents):
        x = zeros((len(raw_documents), self.model.layer1_size))

        for row, doc in enumerate(raw_documents):
            n = 0

            for token in self.analyzer_func(doc):
                if token in self.model:
                    x[row, :] += self.model[token]
                    n += 1

            if n > 0:
                x[row, :] = x[row, :] / n

        return x
