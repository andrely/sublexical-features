import spacy
from numpy import zeros
from numpy.linalg import norm
from sklearn.base import TransformerMixin, BaseEstimator


NLP = spacy.en.English()

def _preprocess(text):
    return [token.text.lower().strip() for token in NLP(text) if token.text.strip() != ""]


class Word2VecVectorizer(TransformerMixin, BaseEstimator):
    def __init__(self, w2v_model=None, vocab=None):
        self._w2v_model = w2v_model
        self._vocab = vocab
        self.dim = w2v_model.syn0.shape[1]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        result = zeros((len(X), self.dim))

        for i, sent in enumerate(X):
            if not isinstance(sent, unicode):
                sent = unicode(sent)

            vect = zeros(self.dim)
            count = 0

            for token in _preprocess(sent):
                if (not self._vocab) or self._vocab.token2id.has_key(token):
                    if self._w2v_model.vocab.has_key(token):
                        vect += self._w2v_model.syn0[self._w2v_model.vocab[token].index, :]
                        count += 1

            v_norm = norm(vect)

            if abs(v_norm) > .00001:
                vect = vect / v_norm

            result[i, :] = vect

        return result
