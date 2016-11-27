import logging
from optparse import OptionParser
import os
import sys

import numpy


cur_path, _ = os.path.split(__file__)
sys.path.append(os.path.join(cur_path, '..', 'Corpora'))
sys.path.append(os.path.join(cur_path, '..', 'Experiments'))
sys.path.append(os.path.join(cur_path, '..', 'BrownClustering'))

from experiment_support.experiment_runner import FilteredSequence
from shared_corpora.pan import PanAPSequence, pan_ap_corpus_path, build_pan_ap_db

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = OptionParser()
    parser.add_option("-f", "--database-file", default='default.db')
    parser.add_option("-s", "--sample")
    parser.add_option("-c", "--corpus-path", default=pan_ap_corpus_path)
    opts, args = parser.parse_args()

    db_fn = opts.database_file
    corpus_path = opts.corpus_path

    sample = None
    if opts.sample:
        sample = int(opts.sample)

    seq = PanAPSequence(corpus_path)

    logging.info("Building PAN AP database %s" % db_fn)

    if sample:
        logging.info("Sampling %d conversations" % sample)
        seq = FilteredSequence(seq, numpy.random.choice(xrange(len(seq)), sample))

    build_pan_ap_db(db_fn, seq)
