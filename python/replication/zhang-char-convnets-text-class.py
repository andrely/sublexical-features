import logging
import os
from argparse import ArgumentParser
from multiprocessing import cpu_count

import numpy as np
from numpy import sum
from numpy.random.mtrand import choice
from pandas import concat
from pandas.core.frame import DataFrame
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics.classification import accuracy_score
from sklearn.pipeline import Pipeline

from sublexical_semantics.data.ag_news import ag_news_dataset
from sublexical_semantics.data.dbpedia_ontology import dbpedia_ontology_dataset


def stratified_sample(df, size=None, target_col=None):
    if not size or not target_col:
        raise ValueError

    target_counts = df[target_col].value_counts()
    total = sum(target_counts.values)
    stratified_counts = zip(target_counts.index.values, np.round(target_counts.values * size / total).astype(np.int))
    sampled_dfs = [df[df[target_col] == target].iloc[choice(target_counts[target], c, replace=False)]
                   for target, c in stratified_counts]

    return concat(sampled_dfs)


def dbpedia_bwords(sample=None, n_procs=None):
    if not n_procs:
        n_procs = cpu_count()

    df = get_dbpedia_data(size=sample)

    input = [' '.join([t, a]) for t, a in zip(df.title, df.abstract)]
    target = df.category

    if sample:
        test_size = int(round(np.sum(5000*df.category.value_counts().values/45000)))
    else:
        test_size = 5000*14

    X, X_, y, y_ = train_test_split(input, target, stratify=target, test_size=test_size)

    model = Pipeline([('vect', CountVectorizer(max_features=50000)), ('cls', LogisticRegression())])
    grid = GridSearchCV(model, {'cls__C': [.0001, .0003, .001, .003, .01, .03, .1, .3, 1, 3, 10, 30, 100]},
                        n_jobs=n_procs, verbose=1, cv=5)
    grid.fit(X, y)

    print(accuracy_score(y_, grid.best_estimator_.predict(X_)))


def agnews_bwords(sample=None, n_procs=None):
    if not n_procs:
        n_procs = cpu_count()

    df = get_agnews_data(size=sample)

    input = [' '.join([title, descr]) for title, descr in zip(df.title.values, df.description.values)]
    target = df.category

    if sample:
        test_size = int(round(np.sum(2000*df.category.value_counts().values/32000)))
    else:
        test_size = 2000*4

    X, X_, y, y_ = train_test_split(input, target, stratify=target, test_size=test_size)

    model = Pipeline([('vect', CountVectorizer(max_features=50000)), ('cls', LogisticRegression())])
    grid = GridSearchCV(model, {'cls__C': [.0001, .0003, .001, .003, .01, .03, .1, .3, 1, 3, 10, 30, 100]},
                        n_jobs=n_procs, verbose=1, cv=5)
    grid.fit(X, y)

    print(accuracy_score(y_, grid.best_estimator_.predict(X_)))


def get_dbpedia_data(size=None):
    logging.info("Reading DBpedia dataset, using 45k samples")
    df = dbpedia_ontology_dataset('../../data/dbpedia-ontology/')

    top = df.category.value_counts()

    df = concat([df[df.category == cat].iloc[np.random.choice(count, size=min(count, 45000), replace=False)]
                 for cat, count in zip(top.index, top.values)])

    if size:
        logging.info("Using sample of %d data points" % size)

        df = stratified_sample(df, size=size, target_col='category')

    return df


def get_agnews_data(size=None):
    logging.info("Reading AG news dataset, using 32k samples from top 4 categories")
    df = DataFrame(data=ag_news_dataset(os.path.join('..', '..', 'data', 'AG-corpus')))

    top = df.category.value_counts()[0:4]

    df = concat([df[df.category == cat].iloc[np.random.choice(count, size=32000, replace=False)]
                 for cat, count in zip(top.index, top.values)])

    if size:
        logging.info("Using sample of %d data points" % size)

        df = stratified_sample(df, size=size, target_col='category')

    return df


def dbpedia_summary():
    df, x, y = get_dbpedia_data()

    print('Documents: %d' % len(x))
    print(df.category.value_counts())


def agnews_summary():
    df = get_agnews_data()

    print('Documents: %d' % len(df))
    print(df.category.value_counts())


def main():
    parser = ArgumentParser()
    parser.add_argument("-m", "--method")
    parser.add_argument("-d", "--dataset")
    parser.add_argument("-u", "--summary", default=False, action='store_true')
    parser.add_argument("-s", "--sample", default=None)
    parser.add_argument("-p", "--n-procs", default=0, type=int)
    opts = parser.parse_args()

    method = opts.method
    dataset = opts.dataset
    summary = opts.summary
    sample = int(opts.sample) if opts.sample else None
    n_procs = cpu_count() if opts.n_procs == 0 else opts.n_procs

    logging.info("using %d processors" % n_procs)

    if dataset == 'dbpedia' and summary:
        dbpedia_summary()
    if dataset == 'agnews' and summary:
        agnews_summary()

    if dataset == 'dbpedia' and method == 'bagwords':
        dbpedia_bwords(sample=sample, n_procs=n_procs)
    elif dataset == 'agnews' and method == 'bagwords':
        agnews_bwords(sample=sample, n_procs=n_procs)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()
