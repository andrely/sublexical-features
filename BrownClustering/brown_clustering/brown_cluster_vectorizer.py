from scipy.sparse import csr_matrix

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import VectorizerMixin

from brown_clustering.cluster_index import ClusterIndex


class BrownClusterVectorizer(BaseEstimator, VectorizerMixin):
    def __init__(self, cluster_fn):
        self.cluster_fn = cluster_fn

        self.cluster_index = None

        self.analyzer = 'word'
        self.preprocessor = None
        self.strip_accents = 'unicode'
        self.lowercase = True
        self.stop_words = None
        self.tokenizer = None
        self.token_pattern = r"(?u)\b\w\w+\b"
        self.input = None
        self.ngram_range = (1, 1)
        self.encoding = 'utf-8'
        self.decode_error = 'strict'

    def fit(self, raw_documents, y=None):
        with open(self.cluster_fn) as f:
            self.cluster_index = ClusterIndex(f)

        self.analyzer_func = self.build_analyzer()

        return self

    def transform(self, raw_documents):
        x = csr_matrix((len(raw_documents), self.cluster_index.n_cluster))

        for row, doc in enumerate(raw_documents):
            for token in self.analyzer_func(doc):
                idx = self.cluster_index.cluster(token)
                x[row, idx] += 1


        return x
