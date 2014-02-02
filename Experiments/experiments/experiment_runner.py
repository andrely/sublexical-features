from collections import Sequence
import os

from scipy import sparse
from numpy import mean, std
from sklearn.base import BaseEstimator, clone
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from brown_clustering.brown_cluster_vectorizer import BrownClusterVectorizer

from corpora.newsgroups import ArticleSequence, GroupSequence
from experiments.preprocessing import mahoney_clean, sublexicalize


class MahoneyCorpus(object):
    """Iterate over sentences from the "text8" corpus, unzipped from http://mattmahoney.net/dc/text8.zip ."""
    def __init__(self, fname, max_sent_length=1000):
        self.fname = fname
        self.max_sent_length = max_sent_length

    def __iter__(self):
        # the entire corpus is one gigantic line -- there are no sentence marks at all
        # so just split the sequence of tokens arbitrarily: 1 sentence = 1000 tokens
        sentence, rest, max_sentence_length = [], '', self.max_sent_length
        with open(self.fname) as fin:
            while True:
                text = rest + fin.read(8192)  # avoid loading the entire file (=1 line) into RAM
                if text == rest:  # EOF
                    sentence.extend(rest.split()) # return the last chunk of words, too (may be shorter/longer)
                    if sentence:
                        yield sentence
                    break
                last_token = text.rfind(' ')  # the last token may have been split in two... keep it for the next iteration
                words, rest = (text[:last_token].split(), text[last_token:].strip()) if last_token >= 0 else ([], text)
                sentence.extend(words)
                while len(sentence) >= max_sentence_length:
                    yield sentence[:max_sentence_length]
                    sentence = sentence[max_sentence_length:]


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
