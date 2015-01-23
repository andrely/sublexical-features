import logging
import sys
import time

from gensim.corpora.wikicorpus import filter_wiki, tokenize


def main():
    start = time.clock()

    for text in sys.stdin:
        text = filter_wiki(text)
        text = tokenize(text)

        if text:
            sys.stdout.write(' '.join(text))
            sys.stdout.write('\n')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    main()