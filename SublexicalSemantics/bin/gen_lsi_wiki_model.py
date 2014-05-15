import logging
from optparse import OptionParser
import os
import sys

from gensim.corpora import WikiCorpus, Dictionary
from gensim.models import LsiModel, TfidfModel


cur_path, _ = os.path.split(__file__)
# sys.path.append(os.path.join(cur_path, '..', '..', 'Corpora'))
# sys.path.append(os.path.join(cur_path, '..', '..', 'Experiments'))
sys.path.append(os.path.join(cur_path, '..', '..', 'BrownClustering'))

from experiment_support.experiment_runner import SublexicalizedCorpus


def parse_ngram_order(arg_str):
    tokens = arg_str.split(',')

    if len(tokens) == 1:
        order = int(tokens[0])
        return order, order
    elif len(tokens) == 2:
        return int(tokens[0]), int(tokens[1])
    else:
        raise ValueError


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = OptionParser()
    parser.add_option('-f', '--dump-file')
    parser.add_option('-l', '--word-limit', default=None, type=int)
    parser.add_option("-n", "--ngram-order", default="3")
    parser.add_option("-c", "--min-count", default=5, type=int)
    parser.add_option("-m", "--model-file", help="Output model file.")
    parser.add_option("-t", "--num-topics", default=100, type=int)
    parser.add_option("-s", "--scaling")

    opts, args = parser.parse_args()

    if not opts.dump_file:
        raise ValueError('--dump-file argument is required')
    else:
        dump_fn = opts.dump_file

    if not opts.model_file:
        raise ValueError('--model-file argument is required')
    else:
        model_fn = opts.model_file

    word_limit = opts.word_limit

    if word_limit:
        logging.info('Word limit %d' % word_limit)

    order = parse_ngram_order(opts.ngram_order)

    logging.info('Char n-gram order (%d, %d)' % order)
    cutoff = opts.min_count

    num_topics = opts.num_topics

    if opts.scaling == 'tfidf':
        scaling = 'tfidf'
    elif not opts.scaling:
        scaling = None
    else:
        raise ValueError("Only tfidf scaling is supported")

    corpus = SublexicalizedCorpus(WikiCorpus(dump_fn, dictionary=Dictionary()), order=order, word_limit=word_limit)

    voc = Dictionary(corpus)
    voc.filter_extremes(no_below=cutoff)
    voc.compactify()

    bow_corpus = (voc.doc2bow(art) for art in corpus)

    tfidf = None

    if scaling == 'tfidf':
        tfidf = TfidfModel(bow_corpus)
        bow_corpus = (tfidf[voc.doc2bow(art)] for art in corpus)

    model = LsiModel(corpus=bow_corpus, num_topics=num_topics, id2word=voc)
    model.save(model_fn)

    if tfidf:
        tfidf.save(model_fn + '.tfidf')
