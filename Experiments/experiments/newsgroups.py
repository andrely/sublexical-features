import logging
import time

from sklearn import clone

from sklearn.cross_validation import KFold
from sklearn.externals.joblib import Parallel, delayed
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

from brown_clustering.brown_cluster_vectorizer import BrownClusterVectorizer

from corpora.newsgroups import ArticleSequence, newsgroups_corpus_path, GroupSequence, article_count
from experiments.experiment_runner import baseline_pipelines, run_experiment


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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    pipeline = baseline_pipelines()['base_word_nopreproc']

    # run_experiment_prof = MemProf(run_experiment)
    # run_experiment_prof(newsgroups_corpus_path, pipeline, n_folds=5)
    #
    st_time(run_experiment)(newsgroups_corpus_path, pipeline, n_folds=5)
