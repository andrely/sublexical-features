import logging
from optparse import OptionParser
import bz2
import os
import sys

import gensim
from gensim.corpora.wikicorpus import process_article, _extract_pages
from gensim.models.word2vec import Text8Corpus


cur_path, _ = os.path.split(__file__)
sys.path.append(os.path.join(cur_path, '..', 'Corpora'))
sys.path.append(os.path.join(cur_path, '..', 'Experiments'))
sys.path.append(os.path.join(cur_path, '..', 'BrownClustering'))

DEFAULT_ACCURACY_FN='questions-words.txt'
DEFAULT_ACCURACY_CUTOFF=30000


def accuracy(model, accuracy_fn, cutoff):
    result = model.accuracy(accuracy_fn, cutoff)

    total_result = [r for r in result if r['section'] == 'total'][0]

    return total_result['correct'] / float(total_result['incorrect'] + total_result['correct'])

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = OptionParser(usage="%prog -f <WIKIFILE> [options]")
    parser.add_option("-s", "--size", default=100, type=int, help="Word vector size.")
    parser.add_option("-f", "--file", help="Input file (default: a compressed Wiki dump).")
    parser.add_option("-t", "--file-type", default='wiki-bz',
                      help="Corpus file type, bz-wiki, text8 (defaulf: bz-wiki)")
    parser.add_option("-m", "--model-file", help="Output model file.")
    parser.add_option("-p", "--processors", default=1, type=int, help="Number of processors to use.")
    parser.add_option("-w", "--window-size", default=5, type=int, help="Context window size.")
    parser.add_option("-a", "--accuracy", default=None, help="Compute analogy accuracy with file.")
    parser.add_option("-c", "--min-count", default=5, type=int)

    options, args = parser.parse_args()

    if not options.file:
        parser.print_help()
        print "--file argument is required"
        exit(1)

    fn = options.file
    n = options.size
    n_jobs = options.processors
    w = options.window_size
    c = options.min_count

    if not options.model_file:
        model_fn = os.path.splitext(os.path.basename(fn))[0] + "-%d.w2v-gensim" % n
    else:
        model_fn = options.model_file

    if os.path.exists(model_fn):
        logging.error("File already exists, %s" % model_fn)
        exit(1)

    logging.info("Generating word vectors size %d from %s" % (n, fn))

    sent_gen = None

    if options.file_type == 'bz-wiki':
        sent_gen = (process_article((text, None))
                    for title, text in _extract_pages(bz2.BZ2File(fn)))
    elif options.file_type == 'mahoney':
        sent_gen = Text8Corpus(fn)
    else:
        raise ValueError

    model = gensim.models.Word2Vec(sent_gen, workers=n_jobs, window=w, size=n, min_count=c)

    if options.accuracy:
        print accuracy(model, options.accuracy, DEFAULT_ACCURACY_CUTOFF)

    logging.info("Writing model to %s" % model_fn)
    model.save(model_fn)
