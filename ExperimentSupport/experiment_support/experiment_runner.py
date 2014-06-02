import logging
import os
import sys

from gensim.corpora import TextCorpus
from gensim.models import Word2Vec
from scipy import sparse
from numpy import mean, std, zeros, array
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.feature_extraction.text import CountVectorizer, VectorizerMixin
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, MinMaxScaler
from sklearn.svm import LinearSVC

from brown_clustering.brown_cluster_vectorizer import BrownClusterVectorizer
from shared_corpora.newsgroups import ArticleSequence, GroupSequence, newsgroups_corpus_path
from experiment_support.preprocessing import mahoney_clean, sublexicalize


def inclusive_range(a, b=None):
    if not b:
        return range(a + 1)
    else:
        return range(a, b + 1)


class TopicPipeline(BaseEstimator):
    def __init__(self, vectorizer, classifier, multilabel=False, normalizer=None):
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.multilabel = multilabel
        self.normalizer = normalizer

        self.model = None
        self.target_encoder = None

    def fit(self, raw_documents, topics):
        if self.multilabel:
            cls = OneVsRestClassifier(self.classifier)
            self.target_encoder = LabelBinarizer()
        else:
            cls = self.classifier
            self.target_encoder = LabelEncoder()

        y = self.target_encoder.fit_transform(topics)

        if self.normalizer:
            self.model = Pipeline([('vect', self.vectorizer), ('norm', self.normalizer), ('cls', cls)])
        else:
            self.model = Pipeline([('vect', self.vectorizer), ('cls', cls)])
        self.model.fit(raw_documents, y)

        return self

    def predict(self, raw_documents):
        y = self.model.predict(raw_documents)
        topics = self.target_encoder.inverse_transform(y)

        return topics


    def score(self, raw_documents, topics):
        if self.multilabel:
            return f1_score(topics, self.predict(raw_documents))
        else:
            return accuracy_score(topics, self.predict(raw_documents))

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


class SublexicalizedCorpus(TextCorpus):
    def __init__(self, base_corpus, order=3, word_limit=None):
        super(SublexicalizedCorpus, self).__init__()

        if isinstance(order, (list, tuple)):
            self.order = inclusive_range(*order)
        else:
            self.order = [3]

        self.base_corpus = base_corpus
        self.word_limit = word_limit

    def __len__(self):
        return len(self.base_corpus)

    def __iter__(self):
        w_count = 0
        a_count = 0

        for text in self.base_corpus.get_texts():
            w_count += len(text)
            a_count += 1

            sys.stdout.write('.')

            if a_count % 80 == 0:
                sys.stdout.write('\n')

            tokens = []

            for o in self.order:
                tokens += sublexicalize(mahoney_clean(' '.join(text)), order=o, join=False)

            yield tokens

            if self.word_limit and w_count > self.word_limit:
                break


class LimitCorpus(TextCorpus):
    def __init__(self, base_corpus, word_limit):
        super(LimitCorpus, self).__init__()

        self.base_corpus = base_corpus
        self.word_limit = word_limit

    def __len__(self):
        return len(self.base_corpus)

    def __iter__(self):
        w_count = 0
        a_count = 0

        for text in self.base_corpus.get_texts():
            w_count += len(text)
            a_count += 1

            sys.stdout.write('.')

            if a_count % 80 == 0:
                sys.stdout.write('\n')

            yield mahoney_clean(' '.join(text)).split()

            if self.word_limit and w_count > self.word_limit:
                break


def clean_c4(text_str):
    return sublexicalize(mahoney_clean(text_str), order=4)


def clean_c5(text_str):
    return sublexicalize(mahoney_clean(text_str), order=5)


def clean_c6(text_str):
    return sublexicalize(mahoney_clean(text_str), order=6)


def baseline_pipelines(word_repr_path=None):
    if not word_repr_path:
        word_repr_path = os.getcwd()

    return {
        'base_word': TopicPipeline(CountVectorizer(max_features=1000,
                                                   decode_error='ignore',
                                                   strip_accents='unicode',
                                                   preprocessor=mahoney_clean), MultinomialNB()),
        'base_word_nopreproc': TopicPipeline(CountVectorizer(max_features=1000,
                                                   decode_error='ignore',
                                                   strip_accents='unicode',
                                                   preprocessor=mahoney_clean), MultinomialNB()),
        'base_c4': TopicPipeline(CountVectorizer(max_features=1000,
                                                 decode_error='ignore',
                                                 strip_accents='unicode',
                                                 preprocessor=clean_c4), MultinomialNB()),
        'base_c5': TopicPipeline(CountVectorizer(max_features=1000,
                                                 decode_error='ignore',
                                                 strip_accents='unicode',
                                                 preprocessor=clean_c5), MultinomialNB()),
        'base_c6': TopicPipeline(CountVectorizer(max_features=1000,
                                                 decode_error='ignore',
                                                 strip_accents='unicode',
                                                 preprocessor=clean_c6), MultinomialNB()),
        'base_mixed': TopicPipeline(MultiVectorizer([CountVectorizer(max_features=1000,
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
                                                                     preprocessor=clean_c6)]), MultinomialNB()),
        'bcluster_word_metaoptimize':
            TopicPipeline(BrownClusterVectorizer(os.path.join(word_repr_path,
                                                              'brown-rcv1.clean.tokenized-CoNLL03.txt-c3200-freq1.txt')),
                          MultinomialNB()),
        'bcluster_word_wiki8_1024':
            TopicPipeline(BrownClusterVectorizer(os.path.join(word_repr_path,
                                                              'wiki8-c1024')), MultinomialNB()),
        'bcluster_word_wiki8_2048':
            TopicPipeline(BrownClusterVectorizer(os.path.join(word_repr_path,
                                                              'wiki8-c2048')), MultinomialNB()),
        'bcluster_c4_wiki8_1024':
            TopicPipeline(BrownClusterVectorizer(os.path.join(word_repr_path,
                                                              'wiki8-c4-c1024'),
                                                 preprocessor=clean_c4), MultinomialNB()),
        'bcluster_c4_wiki8_2048':
            TopicPipeline(BrownClusterVectorizer(os.path.join(word_repr_path,
                                                              'wiki8-c4-c2048'),
                                                 preprocessor=clean_c4), MultinomialNB()),
        'bcluster_c5_wiki8_1024':
            TopicPipeline(BrownClusterVectorizer(os.path.join(word_repr_path,
                                                              'wiki8-c5-c1024'),
                                                 preprocessor=clean_c5), MultinomialNB()),
        'bcluster_c5_wiki8_2048':
            TopicPipeline(BrownClusterVectorizer(os.path.join(word_repr_path,
                                                              'wiki8-c5-c2048'),
                                                 preprocessor=clean_c5), MultinomialNB()),
        'bcluster_c6_wiki8_1024':
            TopicPipeline(BrownClusterVectorizer(os.path.join(word_repr_path,
                                                              'wiki8-c6-c1024'),
                                                 preprocessor=clean_c6), MultinomialNB()),
        'bcluster_c6_wiki8_2048':
            TopicPipeline(BrownClusterVectorizer(os.path.join(word_repr_path,
                                                              'wiki8-c6-c2048'),
                                                 preprocessor=clean_c6), MultinomialNB()),
        'bcluster_mixed_wiki8_1024':
            TopicPipeline(MultiVectorizer([BrownClusterVectorizer(os.path.join(word_repr_path,
                                                                               'wiki8-c4-c1024'),
                                                                  preprocessor=clean_c4),
                                           BrownClusterVectorizer(os.path.join(word_repr_path,
                                                                               'wiki8-c5-c1024'),
                                                                  preprocessor=clean_c5),
                                           BrownClusterVectorizer(os.path.join(word_repr_path,
                                                                               'wiki8-c6-c1024'),
                                                                  preprocessor=clean_c6)]), MultinomialNB()),
        'bcluster_mixed_wiki8_2048':
            TopicPipeline(MultiVectorizer([BrownClusterVectorizer(os.path.join(word_repr_path,
                                                                               'wiki8-c4-c2048'),
                                                                  preprocessor=clean_c4),
                                           BrownClusterVectorizer(os.path.join(word_repr_path,
                                                                               'wiki8-c5-c2048'),
                                                                  preprocessor=clean_c5),
                                           BrownClusterVectorizer(os.path.join(word_repr_path,
                                                                               'wiki8-c6-c2048'),
                                                                  preprocessor=clean_c6)]), MultinomialNB()),
        'base_svm_word': TopicPipeline(CountVectorizer(max_features=1000,
                                                       decode_error='ignore',
                                                       strip_accents='unicode',
                                                       preprocessor=mahoney_clean), LinearSVC()),
        'base_svm_c4': TopicPipeline(CountVectorizer(max_features=1000,
                                                     decode_error='ignore',
                                                     strip_accents='unicode',
                                                     preprocessor=clean_c4), LinearSVC()),
        'base_svm_c5': TopicPipeline(CountVectorizer(max_features=1000,
                                                     decode_error='ignore',
                                                     strip_accents='unicode',
                                                     preprocessor=clean_c5), LinearSVC()),
        'base_svm_c6': TopicPipeline(CountVectorizer(max_features=1000,
                                                     decode_error='ignore',
                                                     strip_accents='unicode',
                                                     preprocessor=clean_c6), LinearSVC()),
        'sg_word_wiki8_5_1000': TopicPipeline(Word2VecVectorizer(os.path.join(word_repr_path, 'fil8-5-1000.w2v'),
                                                                 preprocessor=mahoney_clean), LinearSVC()),
        'sg_word_wiki8_10_1000': TopicPipeline(Word2VecVectorizer(os.path.join(word_repr_path, 'fil8-10-1000.w2v'),
                                                                  preprocessor=mahoney_clean), LinearSVC()),
        'sg_word_wiki8_5_2000': TopicPipeline(Word2VecVectorizer(os.path.join(word_repr_path, 'fil8-5-2000.w2v'),
                                                                 preprocessor=mahoney_clean), LinearSVC()),
        'sg_word_wiki8_10_2000': TopicPipeline(Word2VecVectorizer(os.path.join(word_repr_path, 'fil8-5-2000.w2v'),
                                                                  preprocessor=mahoney_clean), LinearSVC()),
        'sg_c4_wiki8_25_1000': TopicPipeline(Word2VecVectorizer(os.path.join(word_repr_path, 'fil8-c4-25-1000.w2v'),
                                                                preprocessor=clean_c4), LinearSVC()),
        'sg_c4_wiki8_50_1000': TopicPipeline(Word2VecVectorizer(os.path.join(word_repr_path, 'fil8-c4-50-1000.w2v'),
                                                                preprocessor=clean_c4), LinearSVC()),
        'sg_c4_wiki8_25_2000': TopicPipeline(Word2VecVectorizer(os.path.join(word_repr_path, 'fil8-c4-25-2000.w2v'),
                                                                preprocessor=clean_c4), LinearSVC()),
        'sg_c4_wiki8_50_2000': TopicPipeline(Word2VecVectorizer(os.path.join(word_repr_path, 'fil8-c4-50-2000.w2v'),
                                                                preprocessor=clean_c4), LinearSVC()),
        'sg_c5_wiki8_25_1000': TopicPipeline(Word2VecVectorizer(os.path.join(word_repr_path, 'fil8-c5-25-1000.w2v'),
                                                                preprocessor=clean_c5), LinearSVC()),
        'sg_c5_wiki8_50_1000': TopicPipeline(Word2VecVectorizer(os.path.join(word_repr_path, 'fil8-c5-50-1000.w2v'),
                                                                preprocessor=clean_c5), LinearSVC()),
        'sg_c5_wiki8_25_2000': TopicPipeline(Word2VecVectorizer(os.path.join(word_repr_path, 'fil8-c5-25-2000.w2v'),
                                                                preprocessor=clean_c5), LinearSVC()),
        'sg_c5_wiki8_50_2000': TopicPipeline(Word2VecVectorizer(os.path.join(word_repr_path, 'fil8-c5-50-2000.w2v'),
                                                                preprocessor=clean_c5), LinearSVC()),
        'sg_c6_wiki8_25_1000': TopicPipeline(Word2VecVectorizer(os.path.join(word_repr_path, 'fil8-c6-25-1000.w2v'),
                                                                preprocessor=clean_c6), LinearSVC()),
        'sg_c6_wiki8_50_1000': TopicPipeline(Word2VecVectorizer(os.path.join(word_repr_path, 'fil8-c6-50-1000.w2v'),
                                                                preprocessor=clean_c6), LinearSVC()),
        'sg_c6_wiki8_25_2000': TopicPipeline(Word2VecVectorizer(os.path.join(word_repr_path, 'fil8-c6-25-2000.w2v'),
                                                                preprocessor=clean_c6), LinearSVC()),
        'sg_c6_wiki8_50_2000': TopicPipeline(Word2VecVectorizer(os.path.join(word_repr_path, 'fil8-c6-50-2000.w2v'),
                                                                preprocessor=clean_c6), LinearSVC())
    }


def run_experiment(corpus_path, topic_pipeline, n_folds=10, n_jobs=1, verbose=1):
    articles = array(ArticleSequence(corpus_path, preprocessor=mahoney_clean))
    topics = array(GroupSequence(corpus_path))

    scores = cross_val_score(topic_pipeline, articles, topics, cv=KFold(len(articles), n_folds=n_folds, shuffle=True),
                             n_jobs=n_jobs, verbose=verbose)

    return mean(scores), std(scores), scores


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    articles = ArticleSequence(newsgroups_corpus_path)
    topics = GroupSequence(newsgroups_corpus_path)

    model = TopicPipeline(Word2VecVectorizer('../../fil8-5-1000.w2v'), MultinomialNB(),
                          normalizer=MinMaxScaler())
    scores = cross_val_score(model, articles, topics, verbose=2,
                             cv=KFold(len(articles), n_folds=10, shuffle=True))

    print scores
