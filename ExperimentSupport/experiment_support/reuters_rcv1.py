from itertools import izip
import logging
from math import ceil
from operator import itemgetter
from optparse import OptionParser
import sys

from numpy import mean, vstack
from sklearn import clone
from sklearn.externals.joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer

from shared_corpora.reuters_rcv1 import newsitem_gen, TOP5_TOPICS


def fit_and_score_fold(fold, model, X, y, params=None):
    low, high = fold
    X_train = X[0:low] + X[high:]
    y_train = vstack((y[0:low], y[high:]))
    X_test = X[low:high]
    y_test = y[low:high]

    _model = clone(model)

    if params:
        _model.set_params(**params)

    _model.fit(X_train, y_train)
    pred = _model.predict(X_test)

    return f1_score(y_test, pred)


def text_cross_val_score(model, X, y, folds):
    parallel = Parallel(n_jobs=3, verbose=1, pre_dispatch='2*n_jobs')
    cv_scores = parallel(delayed(fit_and_score_fold)(fold, model, X, y) for fold in folds)

    return cv_scores


def text_grid_search_cv(model, X, y, folds, params):
    parallel = Parallel(n_jobs=3, verbose=1, pre_dispatch='2*n_jobs')
    grid_scores = parallel(delayed(fit_and_score_fold)(fold, model, X, y, params=p) for fold in folds for p in params)

    n_folds = len(folds)
    n_params = len(params)

    grouped_scores = [grid_scores[i*n_folds:(i+1)*n_folds] for i in xrange(n_params)]

    results = sorted([(p, mean(scores), scores) for scores, p in izip(grouped_scores, params)],
                     key=itemgetter(1), reverse=True)

    return results[0]


def make_text_folds(n_docs, n_folds):
    return [(int(ceil((float(fold) / num_folds) * n_docs)),
             int(ceil((float(fold + 1) / num_folds) * n_docs)) - 1) for fold in range(n_folds)]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = OptionParser()
    parser.add_option("-c", "--corpus-path")
    parser.add_option("-p", "--processors", type="int", default=1)
    parser.add_option("-m", "--max-features", type="int", default=2000)

    opts, args = parser.parse_args()

    if not opts.corpus_path:
        print parser.print_usage()
        sys.exit(1)
    else:
        corpus_path = opts.corpus_path

    num_proc = opts.processors
    max_features = opts.max_features

    logging.info("Using corpus path %s" % corpus_path)
    logging.info("Using %d processors" % num_proc)
    logging.info("Using max %d features" % max_features)

    texts = []
    target = []

    topics = set(TOP5_TOPICS)
    num_folds = 10

    doc_count = 0

    logging.info("Collecting articles with top5 topics")

    for newsitem in newsitem_gen(corpus_path):
        if newsitem.has_key('topics') and not topics.isdisjoint(newsitem['topics']):
            doc_count += 1

            texts.append('\n'.join(newsitem['text']))
            target.append(list(topics.intersection(newsitem['topics'])))

            if doc_count % 10000 == 0:
                logging.info("Added %d documents..." % doc_count)
                break

    logging.info("Using %d documents." % doc_count)

    logging.info("Binarizing topics")
    binarizer = LabelBinarizer()
    y = binarizer.fit_transform(target)

    split = int(doc_count*0.8)

    logging.info("Creating train/test split")
    texts_train = texts[0:split]
    texts_test = texts[split:]
    y_train = y[0:split]
    y_test = y[split:]

    folds = make_text_folds(split, num_folds)

    model = Pipeline([('vect', TfidfVectorizer(decode_error='ignore', strip_accents='unicode',
                                               max_df=0.5, min_df=5, sublinear_tf=True)),
                      ('nb', OneVsRestClassifier(MultinomialNB()))])

    # cv_scores = text_cross_val_score(model, texts_train, y_train, folds)
    # print "10-fold cv score: %.04f +/- %.04f" % (mean(cv_scores), std(cv_scores))

    grid = ParameterGrid({'vect__max_features': [1000, 2000]})

    print text_grid_search_cv(model, texts_train, y_train, folds, grid)

    logging.info("Evaluating model")
    model = Pipeline([('vect', TfidfVectorizer(decode_error='ignore', strip_accents='unicode',
                                               max_df=0.5, min_df=5, sublinear_tf=True)),
                      ('nb', OneVsRestClassifier(MultinomialNB()))])
    model.fit(texts_train, y_train)

    pred = model.predict(texts_test)
    score = f1_score(y_test, pred)

    print "Evaluation score: %.04f" % score
