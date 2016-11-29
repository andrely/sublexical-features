import logging
import os
import sys
from argparse import ArgumentParser
from math import floor, ceil
from multiprocessing import cpu_count

import numpy as np
from gensim.corpora import Dictionary
from gensim.models import Word2Vec
from keras.layers import Convolution1D, Flatten, Dense, Dropout
from keras.layers import Embedding, MaxPooling1D
from keras.models import Sequential
from numpy import sum
from numpy.random.mtrand import choice
from pandas import concat
from pandas.core.frame import DataFrame
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics.classification import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing.label import LabelBinarizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib')))

from sublexical_semantics.data.ag_news import ag_news_dataset
from sublexical_semantics.data.dbpedia_ontology import dbpedia_ontology_dataset
from sublexical_semantics.data.preprocessing import DataframeSentences, zhang_ch_tokenization, pad_sentences
from sublexical_semantics.data.sogou_news import sogou_news_dataset, read_categories, add_category
from sublexical_semantics.data.yelp import yelp_reviews_df
from sublexical_semantics.feature_extraction.clustered_embeddings_vectorizer import ClusteredEmbeddingsVectorizer


def stratified_sample(df, size=None, target_col=None):
    if not size or not target_col:
        raise ValueError

    target_counts = df[target_col].value_counts()
    total = sum(target_counts.values)
    stratified_counts = zip(target_counts.index.values, np.round(target_counts.values * size / total).astype(np.int))
    sampled_dfs = [df[df[target_col] == target].iloc[choice(target_counts[target], c, replace=False)]
                   for target, c in stratified_counts]

    return concat(sampled_dfs)


def load_w2v_weights(vocab):
    w2v = Word2Vec.load_word2vec_format('../../models/word2vec-binary/GoogleNews-vectors-negative300.bin',
                                        binary=True)
    emb_matrix = np.zeros((len(vocab) + 1, 300))

    for i in range(len(vocab)):
        word = vocab[i]

        if word in w2v.vocab:
            emb_i = w2v.vocab[word].index
            emb_matrix[i + 1, :] = w2v.syn0[emb_i]

    return emb_matrix


def bag_ngram_grid(n_procs):
    model = Pipeline([('vect', TfidfVectorizer(max_features=500000, ngram_range=(1, 5))),
                      ('cls', LogisticRegression())])


    grid = GridSearchCV(model, {'cls__C': [.0001, .0003, .001, .003, .01, .03, .1, .3, 1, 3, 10, 30, 100]},
                        n_jobs=n_procs, verbose=1, cv=5)

    return grid


def bag_words_grid(n_procs):
    model = Pipeline([('vect', TfidfVectorizer(max_features=50000)), ('cls', LogisticRegression())])
    grid = GridSearchCV(model, {'cls__C': [.0001, .0003, .001, .003, .01, .03, .1, .3, 1, 3, 10, 30, 100]},
                        n_jobs=n_procs, verbose=1, cv=5)

    return grid


def dbpedia_train_test_split(sample=None):
    df = get_dbpedia_data(size=sample)

    input = [' '.join([t, a]) for t, a in zip(df.title, df.abstract)]
    target = df.category

    if sample:
        test_size = int(round(np.sum(5000 * df.category.value_counts().values / 45000)))
    else:
        test_size = 5000 * 14

    return train_test_split(input, target, stratify=target, test_size=test_size)

def dbpedia_bwords(sample=None, n_procs=None):
    if not n_procs:
        n_procs = cpu_count()

    X, X_, y, y_ = dbpedia_train_test_split(sample=sample)

    grid = bag_words_grid(n_procs)
    grid.fit(X, y)

    print(accuracy_score(y_, grid.best_estimator_.predict(X_)), grid.best_params_)


def dbpedia_bngrams(sample=None, n_procs=None):
    if not n_procs:
        n_procs = cpu_count()

    X, X_, y, y_ = dbpedia_train_test_split(sample=sample)

    grid = bag_ngram_grid(n_procs)
    grid.fit(X, y)

    print(accuracy_score(y_, grid.best_estimator_.predict(X_)), grid.best_params_)


def dbpedia_bembmeans(sample=None, n_procs=None):
    if not n_procs:
        n_procs = cpu_count()

    df = get_dbpedia_data(size=sample)

    if sample:
        test_size = int(round(np.sum(5000 * df.category.value_counts().values / 45000)))
    else:
        test_size = 5000 * 14

    split = StratifiedShuffleSplit(df.category, test_size=test_size)
    train_split, test_split = next(iter(split))
    train_df = df.iloc[train_split]
    test_df = df.iloc[test_split]

    train_sents = DataframeSentences(train_df, cols=['title', 'abstract'])
    vect = ClusteredEmbeddingsVectorizer(n_clusters=50000).fit(train_sents)

    train_docs = DataframeSentences(train_df, cols=['title', 'abstract'], flatten=True)
    test_docs = DataframeSentences(test_df, cols=['title', 'abstract'], flatten=True)
    X_train = vect.transform(train_docs)
    y_train = train_df.category
    X_test = vect.transform(test_docs)
    y_test = test_df.category

    model = LogisticRegression()
    grid = GridSearchCV(model, {'C': [.0001, .0003, .001, .003, .01, .03, .1, .3, 1, 3, 10, 30, 100]},
                        n_jobs=n_procs, verbose=1, cv=5)
    grid.fit(X_train, y_train)

    print(accuracy_score(y_test, grid.best_estimator_.predict(X_test)), grid.best_params_)


def dbpedia_convemb(sample=None, n_procs=None):
    if not n_procs:
        n_procs = cpu_count()

    df = get_dbpedia_data(size=sample)

    if sample:
        test_size = int(round(np.sum(5000 * df.category.value_counts().values / 45000)))
    else:
        test_size = 5000 * 14

    split = StratifiedShuffleSplit(df.category, test_size=test_size)
    train_split, test_split = next(iter(split))
    train_df = df.iloc[train_split]
    test_df = df.iloc[test_split]

    train_docs = DataframeSentences(train_df, cols=['title', 'abstract'], flatten=True)
    vocab = Dictionary(train_docs)
    vocab.filter_extremes(keep_n=5000)
    bin = LabelBinarizer()

    x_train = np.array(pad_sentences([[vocab.token2id[tok] + 1 for tok in s if tok in vocab.token2id]
                                      for s in train_docs],
                                     max_length=100, padding_word=0))
    y_train = bin.fit_transform(train_df.category.values)

    test_docs = DataframeSentences(test_df, cols=['title', 'abstract'], flatten=True)
    x_test = np.array(pad_sentences([[vocab.token2id[tok] + 1 for tok in s if tok in vocab.token2id]
                                      for s in test_docs],
                                     max_length=100, padding_word=0))
    y_test = bin.transform(test_df.category.values)

    model = Sequential()
    model.add(Embedding(5001, 300, input_length=100, dropout=.2))
    model.add(Convolution1D(nb_filter=50, filter_length=3, border_mode='valid',
                            activation='relu', subsample_length=1))
    model.add(MaxPooling1D(pool_length=model.output_shape[1]))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(14, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train)

    print(accuracy_score(np.argwhere(y_test)[:,1], model.predict_classes(x_test)))


def dbpedia_convgemb(sample=None, n_procs=None):
    if not n_procs:
        n_procs = cpu_count()

    df = get_dbpedia_data(size=sample)

    if sample:
        test_size = int(round(np.sum(5000 * df.category.value_counts().values / 45000)))
    else:
        test_size = 5000 * 14

    split = StratifiedShuffleSplit(df.category, test_size=test_size)
    train_split, test_split = next(iter(split))
    train_df = df.iloc[train_split]
    test_df = df.iloc[test_split]

    train_docs = DataframeSentences(train_df, cols=['title', 'abstract'], flatten=True)
    vocab = Dictionary(train_docs)
    vocab.filter_extremes(keep_n=5000)
    bin = LabelBinarizer()

    x_train = np.array(pad_sentences([[vocab.token2id[tok] + 1 for tok in s if tok in vocab.token2id]
                                      for s in train_docs],
                                     max_length=100, padding_word=0))
    y_train = bin.fit_transform(train_df.category.values)

    test_docs = DataframeSentences(test_df, cols=['title', 'abstract'], flatten=True)
    x_test = np.array(pad_sentences([[vocab.token2id[tok] + 1 for tok in s if tok in vocab.token2id]
                                      for s in test_docs],
                                     max_length=100, padding_word=0))
    y_test = bin.transform(test_df.category.values)

    emb_weights = load_w2v_weights(vocab)

    model = Sequential()
    model.add(Embedding(5001, 300, input_length=100, dropout=.2, weights=[emb_weights], trainable=False))
    model.add(Convolution1D(nb_filter=50, filter_length=3, border_mode='valid',
                            activation='relu', subsample_length=1))
    model.add(MaxPooling1D(pool_length=model.output_shape[1]))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(14, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train)

    print(accuracy_score(np.argwhere(y_test)[:,1], model.predict_classes(x_test)))


def dbpedia_smallwordconv(sample=None, n_procs=None):
    if not n_procs:
        n_procs = cpu_count()

    df = get_dbpedia_data(size=sample)

    if sample:
        test_size = int(round(np.sum(5000 * df.category.value_counts().values / 45000)))
    else:
        test_size = 5000 * 14

    logging.info('creating train test split ...')
    split = StratifiedShuffleSplit(df.category, test_size=test_size)
    train_split, test_split = next(iter(split))
    train_df = df.iloc[train_split]
    test_df = df.iloc[test_split]

    logging.info('preprocessing, padding and binarizing data ...')
    train_docs = DataframeSentences(train_df, cols=['title', 'abstract'], flatten=True)
    vocab = Dictionary(train_docs)
    vocab.filter_extremes(keep_n=5000)
    bin = LabelBinarizer()

    x_train = np.array(pad_sentences([[vocab.token2id[tok] + 1 for tok in s if tok in vocab.token2id]
                                      for s in train_docs],
                                     max_length=100, padding_word=0))
    y_train = bin.fit_transform(train_df.category.values)

    test_docs = DataframeSentences(test_df, cols=['title', 'abstract'], flatten=True)
    x_test = np.array(pad_sentences([[vocab.token2id[tok] + 1 for tok in s if tok in vocab.token2id]
                                      for s in test_docs],
                                     max_length=100, padding_word=0))
    y_test = bin.transform(test_df.category.values)

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
    model.add(Dense(14, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])

    model.fit(x_train, y_train, batch_size=32, nb_epoch=5, validation_data=[x_test, y_test])

    print(accuracy_score(np.argwhere(y_test)[:,1], model.predict_classes(x_test)))

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

    grid = bag_words_grid(n_procs)
    grid.fit(X, y)

    print(accuracy_score(y_, grid.best_estimator_.predict(X_)), grid.best_params_)


def agnews_bngrams(sample=None, n_procs=None):
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

    grid = bag_ngram_grid(n_procs)
    grid.fit(X, y)

    print(accuracy_score(y_, grid.best_estimator_.predict(X_)), grid.best_params_)


def agnews_bembmeans(sample=None, n_procs=None):
    if not n_procs:
        n_procs = cpu_count()

    df = get_agnews_data(size=sample)

    if sample:
        test_size = int(round(np.sum(2000*df.category.value_counts().values/32000)))
    else:
        test_size = 2000*4

    split = StratifiedShuffleSplit(df.category, test_size=test_size)
    train_split, test_split = next(iter(split))
    train_df = df.iloc[train_split]
    test_df = df.iloc[test_split]

    train_sents = DataframeSentences(train_df, cols=['title', 'description'])
    vect = ClusteredEmbeddingsVectorizer(n_clusters=50000).fit(train_sents)

    train_docs = DataframeSentences(train_df, cols=['title', 'description'], flatten=True)
    test_docs = DataframeSentences(test_df, cols=['title', 'description'], flatten=True)
    X_train = vect.transform(train_docs)
    y_train = train_df.category
    X_test = vect.transform(test_docs)
    y_test = test_df.category

    model = LogisticRegression()
    grid = GridSearchCV(model, {'C': [.0001, .0003, .001, .003, .01, .03, .1, .3, 1, 3, 10, 30, 100]},
                        n_jobs=n_procs, verbose=1, cv=5)
    grid.fit(X_train, y_train)

    print(accuracy_score(y_test, grid.best_estimator_.predict(X_test)), grid.best_params_)


def yelp_stars_bwords(sample=None, n_procs=None):
    if not n_procs:
        n_procs = cpu_count()

    df = get_yelp_stars_data(size=sample)

    input = df.text
    target = df.stars

    if sample:
        test_size = floor(len(df) * 1./14)
    else:
        test_size = 10000*len(df.stars.unique())

    X, X_, y, y_ = train_test_split(input, target, stratify=target, test_size=test_size)

    grid = bag_words_grid(n_procs)
    grid.fit(X, y)

    print(accuracy_score(y_, grid.best_estimator_.predict(X_)), grid.best_params_)


def yelpstars_bngrams(sample=None, n_procs=None):
    if not n_procs:
        n_procs = cpu_count()

    df = get_yelp_stars_data(size=sample)

    input = df.text
    target = df.stars

    if sample:
        test_size = floor(len(df) * 1./14)
    else:
        test_size = 10000*len(df.stars.unique())

    X, X_, y, y_ = train_test_split(input, target, stratify=target, test_size=test_size)

    grid = bag_ngram_grid(n_procs)
    grid.fit(X, y)

    print(accuracy_score(y_, grid.best_estimator_.predict(X_)), grid.best_params_)


def yelpstars_bembmeans(sample=None, n_procs=None):
    if not n_procs:
        n_procs = cpu_count()

    df = get_yelp_stars_data(size=sample)

    if sample:
        test_size = floor(len(df) * 1./14)
    else:
        test_size = 10000*len(df.stars.unique())

    split = StratifiedShuffleSplit(df.stars, test_size=test_size)
    train_split, test_split = next(iter(split))
    train_df = df.iloc[train_split]
    test_df = df.iloc[test_split]

    train_sents = DataframeSentences(train_df, cols=['text'])
    vect = ClusteredEmbeddingsVectorizer(n_clusters=50000).fit(train_sents)

    train_docs = DataframeSentences(train_df, cols=['text'], flatten=True)
    test_docs = DataframeSentences(test_df, cols=['text'], flatten=True)
    X_train = vect.transform(train_docs)
    y_train = train_df.stars
    X_test = vect.transform(test_docs)
    y_test = test_df.stars

    model = LogisticRegression()
    grid = GridSearchCV(model, {'C': [.0001, .0003, .001, .003, .01, .03, .1, .3, 1, 3, 10, 30, 100]},
                        n_jobs=n_procs, verbose=1, cv=5)
    grid.fit(X_train, y_train)

    print(accuracy_score(y_test, grid.best_estimator_.predict(X_test)), grid.best_params_)


def sogou_bwords(sample=None, n_procs=None):
    if not n_procs:
        n_procs = cpu_count()

    df = get_sogou_data(size=sample)

    input = [' '.join([title, content]) for title, content in zip(df.contenttitle.values, df.content.values)]
    target = df.cat_en

    if sample:
        test_size = int(round(np.sum(12000*df.cat_en.value_counts().values/102000)))
    else:
        test_size = 12000*5

    X, X_, y, y_ = train_test_split(input, target, stratify=target, test_size=test_size)

    grid = bag_words_grid(n_procs)
    grid.fit(X, y)

    print(accuracy_score(y_, grid.best_estimator_.predict(X_)), grid.best_params_)


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


def get_yelp_stars_data(size=None):
    logging.info("Reading Yelp reviews dataset, using 140k samples from each star rating")
    df = yelp_reviews_df(os.path.join('..', '..', 'data', '2016_Dataset_Challenge_Academic_Dataset'))

    df = concat([df[['text', 'stars']][df.stars == rating].sample(140000) for rating in [1.0, 2.0, 3.0, 4.0, 5.0]])

    if size:
        logging.info("Using sample of %d data points" % size)

        df = stratified_sample(df, size=size, target_col='stars')

    return df


def sogou_data_iter(size=None):
    if not size:
        size = 102000

    cat_size = ceil(size / 5.)
    categories = ['sports', 'finance', 'auto', 'entertainment', 'technology']
    counts = {c: 0 for c in categories}
    cat_data = read_categories(os.path.join('..', '..', 'data', 'sogou_news', 'categories_2012.txt'))

    for doc in sogou_news_dataset(os.path.join('..', '..', 'data', 'sogou_news')):
        doc = add_category(doc, cat_data)
        cat = doc.get('cat_en', None)

        if cat in categories and counts[cat] < cat_size:
            doc['content'] = ' '.join(zhang_ch_tokenization(doc['content']))
            doc['contenttitle'] = ' '.join(zhang_ch_tokenization(doc['contenttitle']))

            yield doc

            counts[cat] += 1

            if __builtins__.sum(counts.values()) >= size:
                return


def get_sogou_data(size=None):
    logging.info('Reading Sogou news dataset, 2.7 mill. web articles')
    logging.info('Using first %d data points' % size)

    return DataFrame(data=sogou_data_iter(size=size))


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
    elif dataset == 'dbpedia' and method == 'bagngrams':
        dbpedia_bngrams(sample=sample, n_procs=n_procs)
    elif dataset == 'dbpedia' and method == 'bagembmeans':
        dbpedia_bembmeans(sample=sample, n_procs=n_procs)
    elif dataset == 'dbpedia' and method == 'convemb':
        dbpedia_convemb(sample=sample, n_procs=n_procs)
    elif dataset == 'dbpedia' and method == 'convgemb':
        dbpedia_convgemb(sample=sample, n_procs=n_procs)
    elif dataset == 'dbpedia' and method == 'smallwordconv':
        dbpedia_smallwordconv(sample=sample, n_procs=n_procs)
    elif dataset == 'agnews' and method == 'bagwords':
        agnews_bwords(sample=sample, n_procs=n_procs)
    elif dataset == 'agnews' and method == 'bagngrams':
        agnews_bngrams(sample=sample, n_procs=n_procs)
    elif dataset == 'agnews' and method == 'bagembmeans':
        agnews_bembmeans(sample=sample, n_procs=n_procs)
    elif dataset == 'yelpstars' and method == 'bagwords':
        yelp_stars_bwords(sample=sample, n_procs=n_procs)
    elif dataset == 'yelpstars' and method == 'bagngrams':
        yelpstars_bngrams(sample=sample, n_procs=n_procs)
    elif dataset == 'yelpstars' and method == 'bagembmeans':
        yelpstars_bembmeans(sample=sample, n_procs=n_procs)
    elif dataset == 'sogou' and method == 'bagwords':
        sogou_bwords(sample=sample, n_procs=n_procs)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()
