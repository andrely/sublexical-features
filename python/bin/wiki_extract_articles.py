import json
import logging
import os
import re
import sys
import timeit
from argparse import ArgumentParser

from sublexical_semantics.data.wikipedia import article_gen


def normalize_title(title):
    return re.sub('\\W+', '', title).lower()


def get_bucket_path(out_path, art_bucket):
    path = os.path.join(out_path, art_bucket)

    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError, e:
            logging.error("Could not create path %s, message %s ..." % (path, e.message))

    if not os.path.isdir(path):
        logging.error("Path %s not a directory ..." % path)

    return path

def main():
    parser = ArgumentParser()
    parser.add_argument('-d', '--dump-file')
    parser.add_argument('-o', '--out-dir')
    parser.add_argument('-r', '--repeat', default=False, type=bool)
    parser.add_argument('-l', '--limit', default=None, type=int)

    opts = parser.parse_args()

    dump_fn = opts.dump_file
    out_path = opts.out_dir
    repeat = opts.repeat
    limit = opts.limit

    if not dump_fn or not out_path:
        sys.exit(1)

    count = 0

    for obj in article_gen(dump_fn, num_articles=limit):
        count += 1

        if count % 10000 == 0:
            logging.info('Read %d articles ...' % count)

        title = normalize_title(obj['title'])

        if len(title) == 1:
            art_bucket = title + '_'
        else:
            art_bucket = title[0:2]

        bucket_path = get_bucket_path(out_path, art_bucket)

        fn = os.path.join(bucket_path, '%s.json' % title)

        if repeat or not os.path.exists(fn):
            with open(fn, 'w') as f:
                json.dump(obj, f)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print timeit.Timer(main).timeit(1)
