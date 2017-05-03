import imp
import logging
import os
import sys
from argparse import ArgumentParser

import numpy as np
from gensim.corpora import Dictionary
from keras.layers import Embedding, Convolution1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.models import Sequential
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib')))

from sublexical_semantics.data.nli2013 import nli2013_df
from sublexical_semantics.data.preprocessing import flatten, pad_sentences


def nli2013_train_test_split(dataset_path):
    train_df = nli2013_df(dataset_path, fold='train')
    test_df = nli2013_df(dataset_path, fold='dev')

    return train_df.text.tolist(), train_df.l1.tolist(), test_df.text.tolist(), test_df.l1.tolist()


def bag_words_grid(n_procs):
    model = Pipeline([('vect', TfidfVectorizer(max_features=50000)), ('cls', LogisticRegression())])
    grid = GridSearchCV(model, {'cls__C': [.0001, .0003, .001, .003, .01, .03, .1, .3, 1, 3, 10, 30, 100]},
                        n_jobs=n_procs, verbose=1, cv=5)

    return grid


def small_word_conv(dataset_path):
    docs, y, test_docs, test_y = nli2013_train_test_split(dataset_path)

    logging.info('preprocessing, padding and binarizing data ...')
    docs = [flatten([sent.split() for sent in doc.split('\n') if sent.strip() != '']) for doc in docs]
    test_docs = [flatten([sent.split() for sent in doc.split('\n') if sent.strip() != '']) for doc in test_docs]

    vocab = Dictionary(docs)
    vocab.filter_extremes(keep_n=5000)
    bin = LabelBinarizer()

    x = np.array(pad_sentences([[vocab.token2id[tok] + 1 for tok in s if tok in vocab.token2id]
                                for s in docs],
                               max_length=100, padding_word=0))
    y = bin.fit_transform(y)

    test_x = np.array(pad_sentences([[vocab.token2id[tok] + 1 for tok in s if tok in vocab.token2id]
                                     for s in test_docs],
                                    max_length=100, padding_word=0))
    test_y = bin.transform(test_y)

    logging.info('building model ...')
    model = Sequential()
    model.add(Embedding(5001, 300, input_length=100))
    model.add(Convolution1D(nb_filter=300, filter_length=7, border_mode='valid',
                            activation='relu', subsample_length=1))
    model.add(MaxPooling1D(pool_length=3, stride=1))
    model.add(Convolution1D(nb_filter=300, filter_length=7, border_mode='valid',
                            activation='relu', subsample_length=1))
    model.add(MaxPooling1D(pool_length=3, stride=1))
    model.add(Convolution1D(nb_filter=300, filter_length=3, border_mode='valid',
                            activation='relu', subsample_length=1))
    model.add(Convolution1D(nb_filter=300, filter_length=3, border_mode='valid',
                            activation='relu', subsample_length=1))
    model.add(Convolution1D(nb_filter=300, filter_length=3, border_mode='valid',
                            activation='relu', subsample_length=1))
    model.add(Convolution1D(nb_filter=300, filter_length=3, border_mode='valid',
                            activation='relu', subsample_length=1))
    model.add(MaxPooling1D(pool_length=3, stride=1))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(11, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])

    model.fit(x, y, batch_size=32, nb_epoch=10, validation_data=[test_x, test_y])

    print(accuracy_score(np.argwhere(test_y)[:, 1], model.predict_classes(test_x)))


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
    elif model_type == 'smallconv':
        small_word_conv(dataset_path)
    else:
        raise ValueError


if __name__ == '__main__':
    imp.reload(logging)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    main()
