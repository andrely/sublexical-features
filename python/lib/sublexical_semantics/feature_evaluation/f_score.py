import numpy as np

from scipy.sparse import issparse

# according to formulation in http://ntur.lib.ntu.edu.tw/bitstream/246246/20060927122855476282/1/features.pdf
# Feature Ranking Using Linear SVM (Chang and Lin 2006)


def f_score(X, y):
    pos = X[y == True]
    neg = X[y == False]

    if (issparse(X)):
        xb = np.array(X.sum(axis=0)).flatten()
        xb_pos = np.array(pos.sum(axis=0)).flatten()
        xb_neg = np.array(neg.sum(axis=0)).flatten()
        ss_pos = np.array(np.sum(np.square(pos - xb_pos), axis=0)).flatten()
        ss_neg = np.array(np.sum(np.square(neg - xb_neg), axis=0)).flatten()
    else:
        xb = np.sum(X, axis=0)
        xb_pos = np.sum(pos, axis=0)
        xb_neg = np.sum(neg, axis=0)
        ss_pos = np.sum(np.square(pos - xb_pos), axis=0)
        ss_neg = np.sum(np.square(neg - xb_neg), axis=0)

    n_pos = np.sum(y)
    n_neg = np.sum(np.logical_not(y))

    score = (((xb_pos - xb) ** 2 + (xb_neg - xb) ** 2) / ((1. / n_pos) * ss_pos + (1. / n_neg) * ss_neg))

    return score
