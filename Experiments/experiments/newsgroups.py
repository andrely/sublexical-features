import logging

from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

from brown_clustering.brown_cluster_vectorizer import BrownClusterVectorizer

from corpora.newsgroups import ArticleSequence, newsgroups_corpus_path, GroupSequence, article_count


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

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # print plain_word_counts(newsgroups_corpus_path)
    print bcluster(newsgroups_corpus_path, "../../brown-rcv1.clean.tokenized-CoNLL03.txt-c3200-freq1.txt")
