from bz2 import BZ2File
import logging
from optparse import OptionParser
import sys
import os
import time

from gensim.corpora.wikicorpus import WikiCorpus
from gensim.corpora import Dictionary
from gensim.models import LsiModel, TfidfModel

from experiment_support.experiment_runner import SublexicalizedCorpus
from experiment_support.preprocessing import normalize_whitespace




# Normalizes numbers and hyphens/brackets in words. from al-Rfou 2014
# tag unknown sublexical units to <UNK> ?

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    parser = OptionParser()
    parser.add_option('-f', '--corpus-file')
    parser.add_option('-p', '--parse-procs', default=1, type=int)
    parser.add_option('-s', '--sublexicalize-procs', default=1, type=int)
    parser.add_option('-t', '--tfidf-model')
    parser.add_option('-v', '--vocabulary')
    parser.add_option('-m', '--model-file')
    opts, args = parser.parse_args()

    corpus_fn = opts.corpus_file or sys.exit()
    n_proc_parse = opts.parse_procs
    n_proc_sublex = opts.sublexicalize_procs
    vocab_fn = opts.vocabulary
    tfidf_fn = opts.tfidf_model
    model_fn = opts.model_file or sys.exit()

    with BZ2File(corpus_fn) as f:
        corpus = SublexicalizedCorpus(WikiCorpus(corpus_fn, processes=n_proc_parse, dictionary=Dictionary()),
                                      order=(3, 6), clean_func=normalize_whitespace, n_proc=n_proc_sublex,
                                      create_dictionary=False)

        if vocab_fn and os.path.exists(vocab_fn):
            logging.info("Loading vocabulary from %s" % vocab_fn)
            vocab = Dictionary.load(vocab_fn)
        else:
            logging.info("Creating vocabulary")

            start = time.clock()
            vocab = Dictionary(corpus.get_texts())
            end = time.clock()
            logging.info("Vocabulary created in %d seconds" % (end - start))

            if vocab_fn:
                logging.info("Saving dictionary to %s" % vocab_fn)
                vocab.save(vocab_fn)

        corpus.dictionary = vocab

        corpus.dictionary.filter_extremes(no_below=5, no_above=.8)
        corpus.dictionary.compactify()

        if tfidf_fn and os.path.exists(tfidf_fn):
            logging.info("Reading TF-IDF model from %s" % tfidf_fn)
            tfidf = TfidfModel.load(tfidf_fn)
        else:
            logging.info("creating TF-IDF model")
            tfidf = TfidfModel(corpus)

            if tfidf_fn:
                logging.info("Saving TFF-IDF model to %s" % tfidf_fn)
                tfidf.save(tfidf_fn)

        bow_corpus = (tfidf[art] for art in corpus)

        model = LsiModel(corpus=bow_corpus, num_topics=10, id2word=corpus.dictionary)

        model.save(model_fn)


if __name__ == '__main__':
    main()
