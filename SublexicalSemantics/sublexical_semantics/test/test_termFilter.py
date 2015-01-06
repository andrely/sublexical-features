from StringIO import StringIO
from unittest import TestCase

from sublexical_semantics.term_filter import TermFilter, get_score_map


FREQ_FILE_STR = """###TOTAL###\t45\t10
1\t1\t1\t0.1
2\t2\t1\t0.2
3\t3\t1\t0.3
4\t4\t1\t0.4
5\t5\t1\t0.5
6\t6\t1\t0.6
7\t7\t1\t0.7
8\t8\t1\t0.8
9\t9\t1\t0.9
"""


class TestTermFilter(TestCase):
    def test___getitem__(self):
        term_filter = TermFilter(get_score_map(StringIO(FREQ_FILE_STR), score='idf'), upper=0.8, lower=0.3)

        self.assertFalse(term_filter['ba'])
        self.assertAlmostEqual(term_filter.upper_percentile, 0.74, delta=0.01)
        self.assertAlmostEqual(term_filter.lower_percentile, 0.34, delta=0.01)
        self.assertTrue(term_filter['4'])
        self.assertTrue(term_filter['5'])
        self.assertTrue(term_filter['6'])
        self.assertTrue(term_filter['7'])
        self.assertFalse(term_filter['1'])
        self.assertFalse(term_filter['2'])
        self.assertFalse(term_filter['3'])
        self.assertFalse(term_filter['8'])
        self.assertFalse(term_filter['9'])
