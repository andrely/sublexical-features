from collections import defaultdict


class ClusterDescription(object):
    def __init__(self):
        self.clusters = {}
        self.index = defaultdict(lambda: None)
        self.size = 0

    def cluster_ids(self):
        return self.clusters.keys()

    def add_cluster(self, words):
        self.size += 1
        cluster_id = self.size

        self.clusters[cluster_id] = words

        for word in words:
            if self.index[word]:
                raise RuntimeError

            self.index[word] = cluster_id

    def merge_cluster(self, c1):
        pass