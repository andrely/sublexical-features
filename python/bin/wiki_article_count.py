import logging
import sys
from argparse import ArgumentParser

from sublexical_semantics.data.wikipedia import article_gen


def main():
    parser = ArgumentParser()
    parser.add_argument('-d', '--dump-file')
    parser.add_argument('-p', '--procs', default=1, type=int)
    opts = parser.parse_args()

    dump_fn = opts.dump_file
    parser_procs = opts.procs

    if not dump_fn:
        sys.exit(1)

    count = 0

    for _ in article_gen(dump_fn, parse=False, n_procs=parser_procs):
        count += 1

        if count % 10000 == 0:
            logging.info('Read %d articles ...' % count)

    print count


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()