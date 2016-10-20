import logging
import os
from argparse import ArgumentParser

import spacy
from gensim.models.word2vec import Word2Vec

from sublexical_semantics.data.json_dump import extracted_gen
from sublexical_semantics.data.wikipedia import article_gen


def get_dump_gen(dump_loc, limit=None, n_procs=1):
    if os.path.isdir(dump_loc):
        return extracted_gen(dump_loc, limit=limit)
    else:
        return article_gen(dump_loc, n_procs=n_procs, num_articles=limit)


def sentences(dump_loc, nlp, limit=None, n_procs=1):
    return ([token.text.lower().strip() for token in doc if token.text.strip() != ""]
            for doc in nlp.pipe((art['article.text']
                                 for art in get_dump_gen(dump_loc, limit=limit, n_procs=n_procs)),
                                n_threads=n_procs, parse=False, tag=False, entity=False))

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

    nlp = spacy.en.English()

    model = Word2Vec(size=300, workers=n_procs, max_vocab_size=250000)
    model.build_vocab(sentences(dump_loc, nlp, limit=limit, n_procs=n_procs))
    model.save('%s-vocabonly' % out_fn)
    model.train(sentences(dump_loc, nlp, limit=limit, n_procs=n_procs))

    model.save(out_fn)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    main()