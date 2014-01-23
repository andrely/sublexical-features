import unittest

from experiments.preprocessing import mahoney_clean, sublexicalize


enwiki_input = "'''Anarchism''' originated as a term of abuse first used against early [[working class]] radicals" + \
               " including the [[Diggers]] of the [[English Revolution]] and the sans-culottes " + \
               "of the [[French Revolution]]. Whilst" + \
               " the term is still used in a pejorative way to describe ''any act that used violent means to " + \
               "destroy the organization of society, it has also been taken up as a positive label by " + \
               "self-defined anarchists. In 1998."

fil_output = " anarchism originated as a term of abuse first used against early working class radicals including the " + \
             "diggers of the english revolution and the sans culottes of the french revolution whilst the term is " + \
             "still used in a pejorative way to describe any act that used violent means to destroy the organization " + \
             "of society it has also been taken up as a positive label by self defined anarchists in " + \
             "one nine nine eight "


class TestPreprocessing(unittest.TestCase):
    def test_mahoney_clean(self):
        result = mahoney_clean(enwiki_input)
        self.assertEqual(result, fil_output)

    def test_subexicalize(self):
        result = sublexicalize("abc def ghi", order=3)
        self.assertEquals(result, "abc bc_ c_d _de def ef_ f_g _gh ghi")

        result = sublexicalize("abc def ghi", order=4)
        self.assertEquals(result, "abc_ bc_d c_de _def def_ ef_g f_gh _ghi")

if __name__ == '__main__':
    unittest.main()
