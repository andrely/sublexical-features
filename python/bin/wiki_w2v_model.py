import logging
import os
import sys
from argparse import ArgumentParser

from gensim.models.word2vec import Word2Vec

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'lib'))

from sublexical_semantics.data.cirrus_wikipedia import CirrusDumpIter


def main():
    parser = ArgumentParser()
    parser.add_argument('-d', '--wiki-dump')
    parser.add_argument('-l', '--limit', default=None, type=int)
    parser.add_argument('-p', '--num-procs', default=1, type=int)
    parser.add_argument('-o', '--out', default='wiki-w2v')
    opts = parser.parse_args()

    dump_loc = opts.wiki_dump
    limit = opts.limit
    n_procs = opts.num_procs
    out_fn = opts.out

    model = Word2Vec(size=300, workers=n_procs, max_vocab_size=250000)
    articles = CirrusDumpIter(dump_loc, limit=limit)

    model.build_vocab(articles)
    model.save('%s-vocabonly' % out_fn)
    model.train(articles)

    model.save(out_fn)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    main()