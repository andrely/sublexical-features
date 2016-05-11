import logging
import os
from argparse import ArgumentParser

import spacy
from gensim.corpora.dictionary import Dictionary

from sublexical_semantics.data.json_dump import extracted_gen
from sublexical_semantics.data.wikipedia import article_gen


def get_dump_gen(dump_loc, limit=None, n_procs=1):
    if os.path.isdir(dump_loc):
        return extracted_gen(dump_loc, limit=limit)
    else:
        return article_gen(dump_loc, n_procs=n_procs, num_articles=limit)


def main():
    parser = ArgumentParser()
    parser.add_argument('-d', '--wiki-dump')
    parser.add_argument('-l', '--limit', default=None, type=int)
    parser.add_argument('-p', '--num-procs', default=1, type=int)
    parser.add_argument('-o', '--out', default='vocab')
    opts = parser.parse_args()

    dump_loc = opts.wiki_dump
    limit = opts.limit
    n_procs = opts.num_procs
    out_fn = opts.out

    dump_gen = get_dump_gen(dump_loc, limit=limit)

    nlp = spacy.English()
    vocab = Dictionary(([token.text.lower().strip() for token in doc if token.text.strip() != ""]
                        for doc in nlp.pipe((art['article.text'] for art in dump_gen), n_threads=n_procs,
                                            parse=False, tag=False, entity=False)))

    vocab.save('%s.vocab' % out_fn)
    vocab.save_as_text('%s.txt' % out_fn)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    main()
