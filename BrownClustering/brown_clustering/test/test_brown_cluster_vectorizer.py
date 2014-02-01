from cStringIO import StringIO
import os
from unittest import TestCase

from brown_clustering.brown_cluster_vectorizer import BrownClusterVectorizer
from brown_clustering.cluster_index import _parse_cluster_file


cur_dir, _ = os.path.split(__file__)
data_path = os.path.join(cur_dir, '..', '..', 'data')
CLUSTER_FN = os.path.join(data_path, 'paths')

cluster_input = "0\tthe\t6\n10\tchased\t3\n110\tdog\t2\n1110\tmouse\t2\n1111\tcat\t2\n"
vect_input = ["the cat chased the mouse",
              "the dog chased the cat",
              "the mouse chased the dog"]

class TestBrownClusterVectorizer(TestCase):
    def test_parse_cluster_file(self):
        word_to_cluster, cluster_to_word, freqs = _parse_cluster_file(StringIO(cluster_input))

        self.assertEquals(word_to_cluster['the'], '0')
        self.assertEquals(word_to_cluster['chased'], '10')
        self.assertEquals(word_to_cluster['dog'], '110')
        self.assertEquals(word_to_cluster['mouse'], '1110')
        self.assertEquals(word_to_cluster['cat'], '1111')

        self.assertEquals(cluster_to_word['0'], ['the'])
        self.assertEquals(cluster_to_word['10'], ['chased'])
        self.assertEquals(cluster_to_word['110'], ['dog'])
        self.assertEquals(cluster_to_word['1110'], ['mouse'])
        self.assertEquals(cluster_to_word['1111'], ['cat'])

        self.assertEquals(freqs['the'], 6)
        self.assertEquals(freqs['dog'], 2)
        self.assertEquals(freqs['cat'], 2)
        self.assertEquals(freqs['mouse'], 2)
        self.assertEquals(freqs['chased'], 3)

    def test_fit(self):
        vect = BrownClusterVectorizer(CLUSTER_FN)
        vect.fit(vect_input)
        result = vect.transform(vect_input)

        self.assertEquals(result[0, vect.vocabulary_['0']], 2)
        self.assertEquals(result[0, vect.vocabulary_['10']], 1)
        self.assertEquals(result[0, vect.vocabulary_['1111']], 1)
        self.assertEquals(result[0, vect.vocabulary_['1110']], 1)

        self.assertEquals(result[1, vect.vocabulary_['0']], 2)
        self.assertEquals(result[1, vect.vocabulary_['10']], 1)
        self.assertEquals(result[1, vect.vocabulary_['110']], 1)
        self.assertEquals(result[1, vect.vocabulary_['1111']], 1)

        self.assertEquals(result[2, vect.vocabulary_['0']], 2)
        self.assertEquals(result[2, vect.vocabulary_['10']], 1)
        self.assertEquals(result[2, vect.vocabulary_['110']], 1)
        self.assertEquals(result[2, vect.vocabulary_['1110']], 1)

        self.assertEquals(result.sum(), 15.0)
