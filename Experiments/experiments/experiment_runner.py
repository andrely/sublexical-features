from collections import Sequence

from scipy import sparse
from numpy import mean, std
from sklearn.base import BaseEstimator, clone
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

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


def clean_c4(text_str):
    return sublexicalize(mahoney_clean(text_str), order=4)


def clean_c5(text_str):
    return sublexicalize(mahoney_clean(text_str), order=5)


def clean_c6(text_str):
    return sublexicalize(mahoney_clean(text_str), order=6)


def baseline_pipelines():
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
