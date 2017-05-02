import imp
import logging
import os
from argparse import ArgumentParser

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sublexical_semantics.data.nli2013 import nli2013_df


def nli2013_train_test_split(dataset_path):
    train_df = nli2013_df(dataset_path, fold='train')
    test_df = nli2013_df(dataset_path, fold='dev')

    return train_df.text.tolist(), train_df.l1.tolist(), test_df.text.tolist(), test_df.l1.tolist()


def bag_words_grid(n_procs):
    model = Pipeline([('vect', TfidfVectorizer(max_features=50000)), ('cls', LogisticRegression())])
    grid = GridSearchCV(model, {'cls__C': [.0001, .0003, .001, .003, .01, .03, .1, .3, 1, 3, 10, 30, 100]},
                        n_jobs=n_procs, verbose=1, cv=5)

    return grid


def main():
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset-path', default=os.getcwd())
    parser.add_argument('-p', '--procs', default=0, type=int)
    parser.add_argument('-m', '--model-type', default='lr')
    opts = parser.parse_args()

    dataset_path = opts.dataset_path
    n_procs = opts.procs
    model_type = opts.model_type

    if model_type == 'lr':
        grid = bag_words_grid(n_procs=n_procs)
        x, y, test_x, test_y = nli2013_train_test_split(dataset_path)
        grid.fit(x, y)

        print(accuracy_score(test_y, grid.best_estimator_.predict(test_x)), grid.best_params_)
    else:
        raise ValueError


if __name__ == '__main__':
    imp.reload(logging)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    main()
