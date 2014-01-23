from gensim.corpora import Dictionary
from nltk import FreqDist, MLEProbDist


class Text(object):
    def __init__(self, source, gen_func=lambda x: x):
        self.dictionary = Dictionary([gen_func(source)])
        self.gen_func = gen_func
        self.source = source

        self.word_freqs = FreqDist()
        for word in self.words():
            self.word_freqs.inc(word)

        self.word_dist = MLEProbDist(self.word_freqs)

    def words(self):
        return (self.dictionary.token2id[token] for token in self.gen_func(self.source))

    def clusters(self, cluster_descr):
        return (cluster_descr.index[word] for word in self.words())
