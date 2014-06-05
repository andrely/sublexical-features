import logging
from optparse import OptionParser
import os
import math
import sys

from gensim.corpora import WikiCorpus, Dictionary
from nltk import FreqDist


cur_path, _ = os.path.split(__file__)
sys.path.append(os.path.join(cur_path, '..', '..', 'BrownClustering'))

from experiment_support.experiment_runner import SublexicalizedCorpus
from gen_lsi_wiki_model import parse_ngram_order

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = OptionParser()
    parser.add_option('-f', '--dump-file')
    parser.add_option('-l', '--word-limit', default=None, type=int)
    parser.add_option("-n", "--ngram-order", default="3")
    parser.add_option("-c", "--min-count", default=5, type=int)

    opts, args = parser.parse_args()

    if not opts.dump_file:
        raise ValueError('--dump-file argument is required')
    else:
        dump_fn = opts.dump_file

    word_limit = opts.word_limit

    if word_limit:
        logging.info('Word limit %d' % word_limit)

    order = parse_ngram_order(opts.ngram_order)

    logging.info('Char n-gram order (%d, %d)' % order)
    cutoff = opts.min_count

    corpus = SublexicalizedCorpus(WikiCorpus(dump_fn, dictionary=Dictionary()), order=order, word_limit=word_limit)

    tf = FreqDist()
    df = FreqDist()

    n_docs = 0

    for text in corpus:
        n_docs += 1

        tf.update(text)
        df.update(set(text))

    print "###TOTAL###\t%d\t%d" % (tf.N(), n_docs)

    for token, freq in tf.items():
        if freq < cutoff:
            break

        print "%s\t%d\t%d\t%.6f" % (token, freq, df[token], math.log(float(n_docs)/df[token]))


