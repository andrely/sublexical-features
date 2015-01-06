import unittest

from shared_corpora.sequences import FilteredSequence


class FilteredSequenceTestCase(unittest.TestCase):
    def test_filtered_sequence(self):
        seq = FilteredSequence([1, 2, 3, 4, 5, 6], [1, 3, 5])
        self.assertEquals(len(seq), 3)
        self.assertEquals(seq[0], 2)
        self.assertEquals(seq[1], 4)
        self.assertEquals(seq[2], 6)


if __name__ == '__main__':
    unittest.main()
