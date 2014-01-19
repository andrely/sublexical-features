import unittest

from experiments.preprocessing import mahoney_clean


enwiki_input = "'''Anarchism''' originated as a term of abuse first used against early [[working class]] [[radical]]s" + \
               " including the [[Diggers]] of the [[English Revolution]] and the [[sans-culotte|''sans-culottes'']] " + \
               "of the [[French Revolution]].[http://uk.encarta.msn.com/encyclopedia_761568770/Anarchism.html] Whilst" + \
               " the term is still used in a pejorative way to describe ''&quot;any act that used violent means to " + \
               "destroy the organization of society&quot;''&lt;ref&gt;[http://www.cas.sc.edu/socy/faculty/deflem/" + \
               "zhistorintpolency.html History of International Police Cooperation], from the final protocols of the" + \
               " &quot;International Conference of Rome for the Social Defense Against Anarchists&quot;, " + \
               "1898&lt;/ref&gt;, it has also been taken up as a positive label by self-defined anarchists."

fil_output = " anarchism originated as a term of abuse first used against early working class radicals including the " + \
             "diggers of the english revolution and the sans culottes of the french revolution whilst the term is " + \
             "still used in a pejorative way to describe any act that used violent means to destroy the organization " + \
             "of society it has also been taken up as a positive label by self defined anarchists"


class PreprocessingTestCase(unittest.TestCase):
    def test_mahoney_clean(self):
        result = mahoney_clean(enwiki_input)
        self.assertEqual(result, fil_output)


if __name__ == '__main__':
    unittest.main()
