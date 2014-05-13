import logging
from math import ceil
from optparse import OptionParser
import sys

from numpy import mean, std, vstack
from sklearn.externals.joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer

from shared_corpora.reuters_rcv1 import newsitem_gen, TOP5_TOPICS


def do_fold(fold, num_folds, texts, y, num_docs):
    logging.info("Evaluating fold %d" % fold)
    low = int(ceil((float(fold) / num_folds) * num_docs))
    high = int(ceil((float(fold + 1) / num_folds) * (num_docs + 1)))
    texts_train = texts[0:low] + texts[high:]
    y_train = vstack((y[0:low], y[high:]))
    texts_test = texts[low:high]
    y_test = y[low:high]
    model = Pipeline([('vect', TfidfVectorizer(decode_error='ignore', strip_accents='unicode',
                                               max_df=0.5, min_df=5, sublinear_tf=True)),
                      ('nb', OneVsRestClassifier(MultinomialNB()))])
    model.fit(texts_train, y_train)
    pred = model.predict(texts_test)

    return f1_score(y_test, pred)


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

    parallel = Parallel(n_jobs=3, verbose=1, pre_dispatch='2*n_jobs')
    cv_scores = parallel(delayed(do_fold)(i, num_folds, texts_train, y_train, split) for i in range(num_folds))

    print "10-fold cv score: %.04f +/- %.04f" % (mean(cv_scores), std(cv_scores))

    logging.info("Evaluating model")
    model = Pipeline([('vect', TfidfVectorizer(decode_error='ignore', strip_accents='unicode',
                                               max_df=0.5, min_df=5, sublinear_tf=True)),
                      ('nb', OneVsRestClassifier(MultinomialNB()))])
    model.fit(texts_train, y_train)

    pred = model.predict(texts_test)
    score = f1_score(y_test, pred)

    print "Evaluation score: %.04f" % score
