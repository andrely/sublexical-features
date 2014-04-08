from collections import Sequence, Iterable
import logging

import numpy
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from SharedCorpora.pan import PanAPSequence, pan_ap_corpus_path
from experiments.experiment_runner import FilteredSequence, TopicPipeline


class ItemSequence(Sequence):
    def __init__(self, base_seq, item_getter=lambda x: x):
        self._base_seq = base_seq
        self._item_getter = item_getter

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self._get_index(i) for i in xrange(*index.indices(len(self)))]
        elif isinstance(index, Iterable):
            return [self._get_index(i) for i in index]
        elif isinstance(index, int):
            return self._get_index(index)
        else:
            raise TypeError

    def _get_index(self, index):
        return self._item_getter(self._base_seq.__getitem__(index))

    def __len__(self):
        return len(self._base_seq)


NUM_DOCS = 1000


def gender_getter(ap_item):
    return ap_item[3]


def doc_getter(ap_item):
    return ap_item[5]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    seq = PanAPSequence(pan_ap_corpus_path)

    f_seq = FilteredSequence(seq, numpy.random.choice(xrange(len(seq)), NUM_DOCS, replace=False))

    logging.info("using %d docs" % len(f_seq))

    doc_seq = ItemSequence(f_seq, doc_getter)
    gender_seq = ItemSequence(f_seq, gender_getter)

    model = TopicPipeline(CountVectorizer(max_features=1000, strip_accents='unicode', decode_error='ignore'),
                          MultinomialNB())

    # scores = cross_val_score(model, doc_seq, gender_seq, n_jobs=2, verbose=2)
    scores = cross_val_score(model, doc_seq, gender_seq, n_jobs=2, verbose=2, cv=3)

    print numpy.mean(scores)
    print numpy.std(scores)
    print scores
