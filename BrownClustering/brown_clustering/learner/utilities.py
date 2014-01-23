from math import log
import re

from nltk import FreqDist


test_fn = '/Users/stinky/Work/tools/brown-cluster/input.txt'

word_re = re.compile("^[A-Za-z]+$")


def filtered_tokens(tokens):
    return (token.lower() for token in tokens if word_re.match(token))


def initial_clusters(tokens, n_clusters):
    dist = FreqDist(tokens)

    return dist.keys()[:n_clusters]


def ent_log(x):
    if x == 0.0:
        return 0.0
    else:
        return log(x)
