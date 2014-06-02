from collections import Sequence
import logging
import time
from numpy import array, mean, std

from sklearn import clone

from sklearn.cross_validation import KFold, train_test_split, cross_val_score
from sklearn.externals.joblib import Parallel, delayed
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, Normalizer

# from brown_clustering.brown_cluster_vectorizer import BrownClusterVectorizer
from sklearn.svm import SVC, LinearSVC

from shared_corpora.newsgroups import ArticleSequence, newsgroups_corpus_path, GroupSequence, article_count
from experiments.experiment_runner import baseline_pipelines, run_experiment
from shared_corpora.newsgroups import parse_articles
from shared_corpora.preprocessing import sublexicalize, mahoney_clean, make_preprocessor
from shared_corpora.sequences import ListSequence
from sublexical_semantics.vectorizers import BrownClusterVectorizer, LsiVectorizer, MultiVectorizer, Word2VecVectorizer


def plain_word_counts(corpus_path):
    folds = KFold(article_count, n_folds=10, shuffle=True)

    results = []

    for i, (train_idx, test_idx) in enumerate(folds):
        logging.info("Running fold %d" % i)
        vect = CountVectorizer(max_features=1000, decode_error='ignore', strip_accents='unicode')
        x_train = vect.fit_transform(ArticleSequence(corpus_path, indices=train_idx))

        bin = LabelEncoder()
        y_train = bin.fit_transform(GroupSequence(corpus_path, indices=train_idx))

        x_test = vect.transform(ArticleSequence(corpus_path, indices=test_idx))
        y_test = bin.transform(GroupSequence(corpus_path, indices=test_idx))

        model = MultinomialNB()
        model.fit(x_train, y_train)
        pred = model.predict(x_test)

        score = accuracy_score(y_test, pred)
        logging.info("Completed fold %d with score %.04f" % (i, score))
        results.append(score)

    return results

def bcluster(corpus_path, cluster_fn):
    folds = KFold(article_count, n_folds=10, shuffle=True)

    results = []

    for i, (train_idx, test_idx) in enumerate(folds):
        logging.info("Running fold %d" % i)
        vect = BrownClusterVectorizer(cluster_fn)
        x_train = vect.fit_transform(ArticleSequence(corpus_path, indices=train_idx))

        bin = LabelEncoder()
        y_train = bin.fit_transform(GroupSequence(corpus_path, indices=train_idx))

        x_test = vect.transform(ArticleSequence(corpus_path, indices=test_idx))
        y_test = bin.transform(GroupSequence(corpus_path, indices=test_idx))

        model = MultinomialNB()
        model.fit(x_train, y_train)
        pred = model.predict(x_test)

        score = accuracy_score(y_test, pred)
        logging.info("Completed fold %d with score %.04f" % (i, score))
        results.append(score)

    return results


def run_classification(fold_id, train_idx, test_idx, vectorizer, model):
    corpus_path = newsgroups_corpus_path

    vect = clone(vectorizer)
    x_train = vect.fit_transform(ArticleSequence(corpus_path, indices=train_idx))

    bin = LabelEncoder()
    y_train = bin.fit_transform(GroupSequence(corpus_path, indices=train_idx))

    x_test = vect.transform(ArticleSequence(corpus_path, indices=test_idx))
    y_test = bin.transform(GroupSequence(corpus_path, indices=test_idx))

    clf_model = clone(model)
    clf_model.fit(x_train, y_train)
    pred = clf_model.predict(x_test)

    score = accuracy_score(y_test, pred)

    return score


def run_cv():
    cluster_fn = "../../brown-rcv1.clean.tokenized-CoNLL03.txt-c3200-freq1.txt"

    folds = KFold(article_count, n_folds=10, shuffle=True)

    model = MultinomialNB()
    vectorizer = BrownClusterVectorizer(cluster_fn)
    result = Parallel(n_jobs=2, verbose=2)(delayed(run_classification)(i, train_idx, test_idx, vectorizer,
                                                                       model) for i, (train_idx, test_idx) in enumerate(folds))

    return result


def st_time(func):
    """
        st decorator to calculate the total time of a func
    """

    def st_func(*args, **keyArgs):
        t1 = time.time()
        r = func(*args, **keyArgs)
        t2 = time.time()
        logging.info("Function=%s, Time=%s", func.__name__, t2 - t1)
        return r

    return st_func

def preproc(text):
    return sublexicalize(text, order=4)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # pipeline = baseline_pipelines()['base_word_nopreproc']

    # run_experiment_prof = MemProf(run_experiment)
    # run_experiment_prof(newsgroups_corpus_path, pipeline, n_folds=5)
    #
    # st_time(run_experiment)(newsgroups_corpus_path, pipeline, n_folds=5)

    corpus_path = 'd:\\Corpora\\20news-19997\\20_newsgroups\\'
    texts = []
    targets = []

    for article in parse_articles(corpus_path, fields=['subject', 'body']):
        texts.append('\n'.join((article['subject'], article['body'])))
        targets.append(article['group'])

    texts = ListSequence(texts)
    targets = ListSequence(targets)

    sub_texts = [sublexicalize(text, order=4) for text in texts]
    sub_texts = ListSequence(texts)

    # single word
    # model = Pipeline([('vect', TfidfVectorizer(decode_error='ignore', strip_accents='unicode', max_df=0.5, min_df=5,
    #                                            sublinear_tf=True, ngram_range=(1, 1), analyzer='word')),
    #                   ('cls', LinearSVC())])

    # 4-chargrams
    # model = Pipeline([('vect', TfidfVectorizer(decode_error='ignore', strip_accents='unicode', max_df=0.5, min_df=5,
    #                                            sublinear_tf=True, ngram_range=(4, 4), analyzer='char')),
    #                   ('cls', LinearSVC())])

    # word repr
    # model = Pipeline([('vect', BrownClusterVectorizer('D:\\Work\\sublexical-features\\models\\wcluster\\brown-fil9-c2048-min40\\paths', preprocessor=mahoney_clean)),
    #                   ('tfidf', TfidfTransformer(sublinear_tf=True)),
    #                   ('cls', LinearSVC())])

    # char repr
    # model = Pipeline([('vect', BrownClusterVectorizer('D:\\Work\\sublexical-features\\models\\wcluster\\brown-fil9-n4-c1024-min20\\paths',
    #                                                   preprocessor=preproc)),
    #                   ('tfidf', TfidfTransformer(sublinear_tf=True)),
    #                   ('cls', LinearSVC())])

    # chargram + char repr
    model = Pipeline([('vect', MultiVectorizer([Pipeline([('brown', BrownClusterVectorizer('D:\\Work\\sublexical-features\\models\\wcluster\\brown-fil9-n4-c1024-min20\\paths',
                                                                                           preprocessor=preproc)),
                                                          ('tfidf', TfidfTransformer(sublinear_tf=True, norm=None))]),
                                                TfidfVectorizer(decode_error='ignore', strip_accents='unicode', max_df=0.5, min_df=5,
                                                                sublinear_tf=True, ngram_range=(4, 4), analyzer='char',
                                                                norm=None)])),
                      ('norm', Normalizer()),
                      ('cls', LinearSVC())])

    # texts_train, texts_test, targets_train, targets_test = train_test_split(texts, targets, train_size=0.8)

    scores = cross_val_score(model, texts, targets, cv=10, n_jobs=10, verbose=1)

    print "%.04f +/- %.04f" % (mean(scores), std(scores))

# Single word
# 0.8854 +/- 0.0069

# 4-chargram
# 0.8899 +/- 0.0069

# word brown
# 0.7667 +/- 0.0082

# w2v
# 0.6025 +/- 0.0099

# char brown no idf
# 0.6623 +/- 0.0079

# char + brown
# 0.8882 +/- 0.0061

# char + w2v
# 0.8863 +/- 0.0066