import logging
import sys

from gensim.corpora.wikicorpus import extract_pages


def main():
    texts = ((text, title, pageid) for title, text, pageid
             in extract_pages(sys.stdin))

    for text, title, pageid in texts:
        text.replace('\n', ' ')
        sys.stdout.write(text.encode('utf-8'))
        sys.stdout.write('\n')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    main()