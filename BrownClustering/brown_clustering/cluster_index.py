from collections import defaultdict
import logging


class ClusterIndex(object):
    def __init__(self, file):
        self.word_to_cluster, self.cluster_to_word = _parse_cluster_file(file)

        self.n_cluster = max(self.cluster_to_word.keys()) + 1

    def cluster(self, word):
        return self.word_to_cluster[word]

def _parse_cluster_file(f):
    line_num = 0
    word_to_cluster = {}
    cluster_to_word = defaultdict(lambda: [])

    for line in f.readlines():
        line_num += 1

        tokens = line.strip().split("\t")

        if len(tokens) != 3:
            logging.warn("Couldn't parse line %d" % line_num)
        else:
            c = int(tokens[0], 2)
            word = tokens[1]

            word_to_cluster[word] = c
            cluster_to_word[c] += [word]

    return word_to_cluster, cluster_to_word