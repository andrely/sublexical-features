from gensim.models import Word2Vec
from scipy.sparse import lil_matrix
from sklearn.base import BaseEstimator
from sklearn.cluster.k_means_ import MiniBatchKMeans


class ClusteredEmbeddingsVectorizer(BaseEstimator):
    def __init__(self, embedding_dim=300, n_clusters=500, vocab_cutoff=5):
        self._w2v_model = None
        self._kmeans_model = None
        self._id2cluster = None

        self.embedding_dim = embedding_dim
        self.n_clusters = n_clusters
        self.vocab_cutoff = vocab_cutoff

    def fit(self, sent_docs, y=None):
        self._w2v_model = Word2Vec(sentences=sent_docs, size=self.embedding_dim, min_count=self.vocab_cutoff)
        self._kmeans_model = MiniBatchKMeans(n_clusters=self.n_clusters).fit(self._w2v_model.syn0)
        self._id2cluster = self._kmeans_model.predict(self._w2v_model.syn0)

        return self

    def transform(self, sent_docs):
        v = lil_matrix((len(sent_docs), self._kmeans_model.n_clusters))

        for i, sent in enumerate(sent_docs):
            for token in sent:
                idx = self._word2cluster(token)

                if idx:
                    v[i, idx] += 1.

        return v

    def _word2cluster(self, token):
        try:
            return self._id2cluster[self._w2v_model.vocab[token].index]
        except KeyError:
            return None
