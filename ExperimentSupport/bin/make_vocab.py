from argparse import ArgumentParser
import logging
import os
from shutil import copyfileobj
import sys
import codecs
import bz2

from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel


def main():
    parser = ArgumentParser()
    parser.add_argument('-e', '--encoding')
    parser.add_argument('-o', '--output-file')
    args = parser.parse_args()

    encoding = args.encoding
    output_fn = args.output_file

    if not output_fn:
        sys.exit(-1)

    if encoding:
        sys.stdout = codecs.getwriter(encoding)(sys.stdout)
        sys.stdin = codecs.getreader(encoding)(sys.stdin)

    texts = (line.split() for line in sys.stdin)

    logging.info('Creating vocabulary ...')
    vocab = Dictionary(texts)

    logging.info('Saving vocabulary to %s ...' % (output_fn + '.bz2'))
    vocab.save(output_fn)

    logging.info('Compressing vocabulary ...')

    with open(output_fn, 'rb') as input:
        with bz2.BZ2File(output_fn + '.bz2', 'wb', compresslevel=9) as output:
            copyfileobj(input, output)

    os.remove(output_fn)

    logging.info('Creating IDF model ...')
    tfidf = TfidfModel(dictionary=vocab)

    logging.info('Saving IDF model to %s ...' % (output_fn + '.tfidf.bz2'))
    tfidf.save(output_fn + '.tfidf')

    logging.info('Compressing IDF model ...')

    with open(output_fn + '.tfidf', 'rb') as input:
        with bz2.BZ2File(output_fn + '.tfidf.bz2', 'wb', compresslevel=9) as output:
            copyfileobj(input, output)

    os.remove(output_fn + '.tfidf')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    main()