import logging
from optparse import OptionParser
import sys

from numpy import mean, std
from sklearn.cross_validation import train_test_split, cross_val_score, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer

from shared_corpora.reuters_rcv1 import newsitem_gen, TOP5_TOPICS


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

    logging.info("Creating train/test split")
    texts_train, texts_test, y_train, y_test = train_test_split(texts, y, test_size=0.2)

    logging.info("Running 10-fold cross validation")
    model = Pipeline([('vect', TfidfVectorizer(decode_error='ignore', strip_accents='unicode',
                                               max_df=0.5, min_df=5, sublinear_tf=True)),
                      ('nb', OneVsRestClassifier(MultinomialNB()))])
    cv_scores = cross_val_score(model, texts_train, y_train, n_jobs=num_proc, cv=KFold(len(texts_train), n_folds=10),
                                verbose=1, scoring='f1')

    print "10-fold cv score: %.04f +/- %.04f" % (mean(cv_scores), std(cv_scores))

    logging.info("Evaluating model")
    model = Pipeline([('vect', TfidfVectorizer(decode_error='ignore', strip_accents='unicode',
                                               max_df=0.5, min_df=5, sublinear_tf=True)),
                      ('nb', OneVsRestClassifier(MultinomialNB()))])
    model.fit(texts_train, y_train)

    pred = model.predict(texts_test)
    score = f1_score(y_test, pred)

    print "Evaluation score: %.04f" % score
