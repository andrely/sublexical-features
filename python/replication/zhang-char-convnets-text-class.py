import logging
from argparse import ArgumentParser
from multiprocessing import cpu_count

import numpy as np
from numpy import round, sum
from numpy.random.mtrand import choice
from pandas import concat
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics.classification import accuracy_score
from sklearn.pipeline import Pipeline

from sublexical_semantics.data.dbpedia_ontology import dbpedia_ontology_dataset


def stratified_sample(df, size=None, target_col=None):
    if not size or not target_col:
        raise ValueError

    target_counts = df[target_col].value_counts()
    total = sum(target_counts.values)
    stratified_counts = zip(target_counts.index.values, round(target_counts.values * size / total).astype(np.int))
    sampled_dfs = [df[df[target_col] == target].iloc[choice(target_counts[target], c, replace=False)]
                   for target, c in stratified_counts]

    return concat(sampled_dfs)



def dbpedia_bwords(sample=None, n_procs=None):
    if not n_procs:
        n_procs = cpu_count()

    df, input, target = get_dbpedia_data(size=sample)

    X, X_, y, y_ = train_test_split(input, target, stratify=target, test_size=.2)

    model = Pipeline([('vect', CountVectorizer(max_features=50000)), ('cls', LogisticRegression())])
    grid = GridSearchCV(model, {'cls__C': [.0001, .0003, .001, .003, .01, .03, .1, .3, 1, 3, 10, 30, 100]},
                        n_jobs=n_procs, verbose=1, cv=5)
    grid.fit(X, y)

    print(accuracy_score(y_, grid.best_estimator_.predict(X_)))


def get_dbpedia_data(size=None):
    logging.info("Reading DBpedia dataset")
    df = dbpedia_ontology_dataset('../../data/dbpedia-ontology/')

    if size:
        logging.info("Using sample of %d data points" % size)

        df = stratified_sample(df, size=size, target_col='category')

    input = [' '.join([t, a]) for t, a in zip(df.title, df.abstract)]
    target = df.category

    return df, input, target


def dbpedia_summary():
    df, x, y = get_dbpedia_data()

    print('Documents: %d' % len(x))
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

    logging.info("using %d processos" % n_procs)

    if dataset == 'dbpedia' and summary:
        dbpedia_summary()
    elif dataset == 'dbpedia' and method == 'bagwords':
        dbpedia_bwords(sample=sample, n_procs=n_procs)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()