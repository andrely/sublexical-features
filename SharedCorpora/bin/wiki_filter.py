import logging
import sys
import codecs

from gensim.corpora.wikicorpus import filter_wiki
from gensim.utils import PAT_ALPHABETIC


sys.stdout=codecs.getwriter('utf-8')(sys.stdout)
sys.stdin=codecs.getreader('utf-8')(sys.stdin)


def main():
    for text in sys.stdin:
        text = filter_wiki(text)
        tokens = [match.group() for match in PAT_ALPHABETIC.finditer(text)]

        if tokens:
            text = ' '.join(tokens)
            sys.stdout.write(text + '\n')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    main()
