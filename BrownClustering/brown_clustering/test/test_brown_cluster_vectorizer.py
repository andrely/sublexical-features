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
        word_to_cluster, cluster_to_word = _parse_cluster_file(StringIO(cluster_input))

        self.assertEquals(word_to_cluster['the'], 0)
        self.assertEquals(word_to_cluster['chased'], 2)
        self.assertEquals(word_to_cluster['dog'], 6)
        self.assertEquals(word_to_cluster['mouse'], 14)
        self.assertEquals(word_to_cluster['cat'], 15)

        self.assertEquals(cluster_to_word[0], ['the'])
        self.assertEquals(cluster_to_word[2], ['chased'])
        self.assertEquals(cluster_to_word[6], ['dog'])
        self.assertEquals(cluster_to_word[14], ['mouse'])
        self.assertEquals(cluster_to_word[15], ['cat'])

    def test_fit(self):
        vect = BrownClusterVectorizer(CLUSTER_FN)
        vect.fit(vect_input)
        result = vect.transform(vect_input)

        self.assertEquals(result[0,0], 2)
        self.assertEquals(result[0,2], 1)
        self.assertEquals(result[0,15], 1)
        self.assertEquals(result[0,14], 1)

        self.assertEquals(result[1,0], 2)
        self.assertEquals(result[1,2], 1)
        self.assertEquals(result[1,6], 1)
        self.assertEquals(result[1,15], 1)

        self.assertEquals(result[2,0], 2)
        self.assertEquals(result[2,2], 1)
        self.assertEquals(result[2,6], 1)
        self.assertEquals(result[2,14], 1)

        self.assertEquals(result.sum(), 15.0)
