import unittest

from brown_clustering.utilities import initial_clusters


class UtilitiesTestCase(unittest.TestCase):
    def setUp(self):
        self.test_input = ['the', 'cat', 'chased', 'the', 'mouse',
                           'the', 'dog', 'chased', 'the', 'cat',
                           'the', 'mouse', 'chased', 'the', 'dog']

    def test_initial_clusters(self):
        result = initial_clusters(self.test_input, 3)
        self.assertEqual(result, ['the', 'chased', 'cat'])


if __name__ == '__main__':
    unittest.main()
