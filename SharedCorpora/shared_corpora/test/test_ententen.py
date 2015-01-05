from unittest import TestCase

from ententen import clean_line


class TestEnTenTen(TestCase):
    def test_clean_line(self):
        self.assertEquals('we , human be subsystem in the holistic system view .',
                          clean_line('<s> we , human be subsystem in the holistic system view . </s>'))