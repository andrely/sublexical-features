from nltk import FreqDist, MLEProbDist, bigrams
from numpy import zeros, inf, unravel_index

from cluster_description import ClusterDescription
from utilities import ent_log


def recompute_cluster_dists(text, cluster_descr):
    c_freqs = FreqDist()
    for c in text.clusters(cluster_descr):
        c_freqs.inc(c)
    c_dist = MLEProbDist(c_freqs)

    c_bi_freqs = FreqDist()
    for bi_c in bigrams(text.clusters(cluster_descr)):
        c_bi_freqs.inc(bi_c)
    c_bi_dist = MLEProbDist(c_bi_freqs)

    return c_dist, c_bi_dist


def w(c1, c2, c_dist, c_bi_dist):
    p_c1 = c_dist.prob(c1)
    p_c2 = c_dist.prob(c2)
    p_c1_c2 = c_bi_dist.prob((c1, c2))
    p_c2_c1 = c_bi_dist.prob((c2, c1))

    if p_c1 == 0. or p_c2 == 0.:
        result = 0.
    elif c1 == c2:
        result = p_c1_c2 * ent_log(p_c1_c2 / (p_c1 * p_c2))
    else:
        result = p_c1_c2 * ent_log(p_c1_c2 / (p_c1 * p_c2)) + p_c2_c1 * ent_log(p_c2_c1 / (p_c2 * p_c1))

    return result


def merged_w(c1, c2, d, c_dist, c_bi_dist):
    p_c = c_dist.prob(c1) + c_dist.prob(c2)
    p_d = c_dist.prob(d)
    p_c_d = c_bi_dist.prob((c1, d)) + c_bi_dist.prob((c2, d))
    p_d_c = c_bi_dist.prob((d, c1)) + c_bi_dist.prob((d, c2))

    return p_c_d * ent_log(p_c_d / (p_c * p_d)) + p_d_c * ent_log(p_d_c / p_d * p_c)


def l(c1, c2, c_dist, c_bi_dist, cluster_descr):
    clusters = cluster_descr.cluster_ids()

    return sum([merged_w(c1, c2, d, c_dist, c_bi_dist) - w(c1, d, c_dist, c_bi_dist) - w(c2, d, c_dist, c_bi_dist)
                for d in clusters])


class Learner(object):
    def __init__(self, text, c=3):
        self.text = text
        self.cluster_descr = ClusterDescription()

        for word in self.text.word_freqs.keys()[:c]:
            self.cluster_descr.add_cluster([word])

        self.c_dist, self.c_bi_dist = recompute_cluster_dists(text, self.cluster_descr)

        self.w_m = self.recompute_weights()
        self.l_m = None

        self.remaining_words = self.text.word_freqs.keys()[c:]

    def recompute_weights(self):
        c = self.cluster_descr.size

        w_m = zeros((c, c))
        for i in xrange(c):
            for j in xrange(c):
                w_m[i, j] = w(i + 1, j + 1, self.c_dist, self.c_bi_dist)

        return w_m

    def recompute_deltas(self):
        c = self.cluster_descr.size

        l_m = zeros((c, c))
        for i in xrange(c):
            for j in xrange(c):
                if j > i:
                    l_m[i, j] = l(i + 1, j + 1, self.c_dist, self.c_bi_dist, self.cluster_descr)
                else:
                    l_m[i, j] = -inf

        return l_m

    def add_cluster(self):
        next_word = self.remaining_words.pop()

        self.cluster_descr.add_cluster([next_word])

        self.c_dist, self.c_bi_dist = recompute_cluster_dists(self.text, self.cluster_descr)
        self.w_m = self.recompute_weights()
        self.l_m = self.recompute_deltas()

        c1, c2 = unravel_index(self.l_m.argmax(), self.l_m.shape)
