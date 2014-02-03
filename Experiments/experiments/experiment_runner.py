from collections import Sequence
import os

from gensim.models import Word2Vec

from scipy import sparse
from numpy import mean, std, zeros
from sklearn.base import BaseEstimator, clone, TransformerMixin
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.feature_extraction.text import CountVectorizer, VectorizerMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from brown_clustering.brown_cluster_vectorizer import BrownClusterVectorizer

from corpora.newsgroups import ArticleSequence, GroupSequence
from experiments.preprocessing import mahoney_clean, sublexicalize


class FilteredSequence(Sequence):
    def __init__(self, base_sequence, included_indices):
        self.base_sequence = base_sequence
        self.included_indices = included_indices

    def __len__(self):
        return len(self.included_indices)

    def __getitem__(self, index):
        return self.base_sequence[self.included_indices[index]]


class TextPipeline(BaseEstimator):
    def __init__(self, vectorizer, classifier, target_encoder):
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.target_encoder = target_encoder


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


class Word2VecVectorizer(BaseEstimator, TransformerMixin, VectorizerMixin):
    def __init__(self, w2v_fn, preprocessor=None):
        self.w2v_fn = w2v_fn

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
        self.decode_error = 'strict'

        self.analyzer_func = None

    def fit(self, x, y=None):
        self.analyzer_func = self.build_analyzer()

        self.model = Word2Vec.load(self.w2v_fn)

        return self

    def transform(self, raw_documents, y=None):
        p = self.model.layer1_size
        n = len(raw_documents)

        x = zeros((n, p))

        for row, doc in enumerate(raw_documents):
            for token in self.analyzer_func(doc):
                if token in self.model:
                    x[row,:] += self.model[token]

        return x


def clean_c4(text_str):
    return sublexicalize(mahoney_clean(text_str), order=4)


def clean_c5(text_str):
    return sublexicalize(mahoney_clean(text_str), order=5)


def clean_c6(text_str):
    return sublexicalize(mahoney_clean(text_str), order=6)


def baseline_pipelines(brown_cluster_path=None):
    if not brown_cluster_path:
        brown_cluster_path = os.getcwd()

    return {
        'base_word': TextPipeline(CountVectorizer(max_features=1000,
                                                  decode_error='ignore',
                                                  strip_accents='unicode',
                                                  preprocessor=mahoney_clean),
                                  MultinomialNB(),
                                  LabelEncoder()),
        'base_c4': TextPipeline(CountVectorizer(max_features=1000,
                                                decode_error='ignore',
                                                strip_accents='unicode',
                                                preprocessor=clean_c4),
                                MultinomialNB(),
                                LabelEncoder()),
        'base_c5': TextPipeline(CountVectorizer(max_features=1000,
                                                decode_error='ignore',
                                                strip_accents='unicode',
                                                preprocessor=clean_c5),
                                MultinomialNB(),
                                LabelEncoder()),
        'base_c6': TextPipeline(CountVectorizer(max_features=1000,
                                                decode_error='ignore',
                                                strip_accents='unicode',
                                                preprocessor=clean_c6),
                                MultinomialNB(),
                                LabelEncoder()),
        'base_mixed': TextPipeline(MultiVectorizer([CountVectorizer(max_features=1000,
                                                                    decode_error='ignore',
                                                                    strip_accents='unicode',
                                                                    preprocessor=clean_c4),
                                                    CountVectorizer(max_features=1000,
                                                                    decode_error='ignore',
                                                                    strip_accents='unicode',
                                                                    preprocessor=clean_c5),
                                                    CountVectorizer(max_features=1000,
                                                                    decode_error='ignore',
                                                                    strip_accents='unicode',
                                                                    preprocessor=clean_c6)]),
                                   MultinomialNB(),
                                   LabelEncoder()),
        'bcluster_word_metaoptimize':
            TextPipeline(BrownClusterVectorizer(os.path.join(brown_cluster_path,
                                                             'brown-rcv1.clean.tokenized-CoNLL03.txt-c3200-freq1.txt')),
                         MultinomialNB(),
                         LabelEncoder()),
        'bcluster_word_wiki8_1024':
            TextPipeline(BrownClusterVectorizer(os.path.join(brown_cluster_path,
                                                             'wiki8-c1024')),
                         MultinomialNB(),
                         LabelEncoder()),
        'bcluster_word_wiki8_2048':
            TextPipeline(BrownClusterVectorizer(os.path.join(brown_cluster_path,
                                                             'wiki8-c2048')),
                         MultinomialNB(),
                         LabelEncoder()),
        'bcluster_c4_wiki8_1024':
            TextPipeline(BrownClusterVectorizer(os.path.join(brown_cluster_path,
                                                             'wiki8-c4-c1024'),
                                                preprocessor=clean_c4),
                         MultinomialNB(),
                         LabelEncoder()),
        'bcluster_c4_wiki8_2048':
            TextPipeline(BrownClusterVectorizer(os.path.join(brown_cluster_path,
                                                             'wiki8-c4-c2048'),
                                                preprocessor=clean_c4),
                         MultinomialNB(),
                         LabelEncoder()),
        'bcluster_c5_wiki8_1024':
            TextPipeline(BrownClusterVectorizer(os.path.join(brown_cluster_path,
                                                             'wiki8-c5-c1024'),
                                                preprocessor=clean_c5),
                         MultinomialNB(),
                         LabelEncoder()),
        'bcluster_c5_wiki8_2048':
            TextPipeline(BrownClusterVectorizer(os.path.join(brown_cluster_path,
                                                             'wiki8-c5-c2048'),
                                                preprocessor=clean_c5),
                         MultinomialNB(),
                         LabelEncoder()),
        'bcluster_c6_wiki8_1024':
            TextPipeline(BrownClusterVectorizer(os.path.join(brown_cluster_path,
                                                             'wiki8-c6-c1024'),
                                                preprocessor=clean_c6),
                         MultinomialNB(),
                         LabelEncoder()),
        'bcluster_c6_wiki8_2048':
            TextPipeline(BrownClusterVectorizer(os.path.join(brown_cluster_path,
                                                             'wiki8-c6-c2048'),
                                                preprocessor=clean_c6),
                         MultinomialNB(),
                         LabelEncoder()),
        'bcluster_mixed_wiki8_1024':
            TextPipeline(MultiVectorizer([BrownClusterVectorizer(os.path.join(brown_cluster_path,
                                                                              'wiki8-c4-c1024'),
                                                                 preprocessor=clean_c4),
                                          BrownClusterVectorizer(os.path.join(brown_cluster_path,
                                                                              'wiki8-c5-c1024'),
                                                                 preprocessor=clean_c5),
                                          BrownClusterVectorizer(os.path.join(brown_cluster_path,
                                                                              'wiki8-c6-c1024'),
                                                                 preprocessor=clean_c6)]),
                         MultinomialNB(),
                         LabelEncoder()),
        'bcluster_mixed_wiki8_2048':
            TextPipeline(MultiVectorizer([BrownClusterVectorizer(os.path.join(brown_cluster_path,
                                                                              'wiki8-c4-c2048'),
                                                                 preprocessor=clean_c4),
                                          BrownClusterVectorizer(os.path.join(brown_cluster_path,
                                                                              'wiki8-c5-c2048'),
                                                                 preprocessor=clean_c5),
                                          BrownClusterVectorizer(os.path.join(brown_cluster_path,
                                                                              'wiki8-c6-c2048'),
                                                                 preprocessor=clean_c6)]),
                         MultinomialNB(),
                         LabelEncoder())
    }


def run_experiment(corpus_path, text_pipeline, n_folds=10, n_jobs=1, verbose=1):
    x = ArticleSequence(corpus_path)
    enc = clone(text_pipeline.target_encoder)
    y = enc.fit_transform(GroupSequence(corpus_path))

    model = Pipeline([('vect', clone(text_pipeline.vectorizer)), ('cls', clone(text_pipeline.classifier))])

    scores = cross_val_score(model, x, y, cv=KFold(len(x), n_folds=n_folds, shuffle=True),
                             n_jobs=n_jobs, verbose=verbose)

    return mean(scores), std(scores), scores
