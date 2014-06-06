import logging
from optparse import OptionParser
import os
import sys

from sklearn.datasets import fetch_20newsgroups
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from experiment_support.preprocessing import clean_c35


cur_path, _ = os.path.split(__file__)
sys.path.append(os.path.join(cur_path, '..', '..', 'SublexicalSemantics'))

from sublexical_semantics.vectorizers import LsiVectorizer


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = OptionParser()
    parser.add_option("-m", "--model-path", help="Output model file.")
    parser.add_option("-p", "--processors", default=1, type=int)

    opts, args = parser.parse_args()

    if not opts.model_path:
        raise ValueError('--model-path argument is required')
    else:
        model_path = opts.model_path

    n_jobs=opts.processors

    categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']

    svm_grid = {'svm__C': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]}

    vect_args = {'min_df': 5, 'max_df': 0.5, 'sublinear_tf': True, 'preprocessor': clean_c35}

    train_4 = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'),
                                 categories=categories)
    test_4 = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'),
                                categories=categories)

    train_all = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    test_all = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

    for n_cat in [4, 20]:
        for model_fn in ['lsi-12M-c100-n35-20', 'lsi-12M-c200-n35-20', 'lsi-12M-c500-n35-20',
                         'lsi-12M-c1000-n35-20', 'lsi-12M-3-5-20', 'lsi-12M-c5000-n35-20']:
            if n_cat == 4:
                train, test = train_4, test_4
            else:
                train, test = train_all, test_all


            model = Pipeline([('vect', LsiVectorizer(os.path.join(model_path, model_fn),
                                                     preprocessor=clean_c35)),
                              ('svm', LinearSVC())])
            grid = GridSearchCV(model, svm_grid, n_jobs=n_jobs, verbose=1, cv=10)
            grid.fit(train.data, train.target)

            model = Pipeline([('vect', LsiVectorizer(os.path.join(model_path, model_fn),
                                                     preprocessor=clean_c35)),
                              ('svm', LinearSVC())])
            model.set_params(**grid.best_params_)
            model.fit(train.data, train.target)

            pred = model.predict(test.data)

            score = f1_score(test.target, pred)

            print "F-score %.4f over %d categories using model %s." % (score, n_cat, model_fn)
