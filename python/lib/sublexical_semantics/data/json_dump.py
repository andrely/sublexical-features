import json
import logging
import os.path
from random import shuffle


def get_fns(dirs):
    for dir in dirs:
        if os.path.isdir(dir):
            for fn in os.listdir(dir):
                full_fn = os.path.join(dir, fn)
                _, ext = os.path.splitext(fn)

                if ext == '.json':
                    yield full_fn


def extracted_gen(path, limit=None, shuffle_dirs=False):
    counter = 0

    dirs = [os.path.join(path, dir) for dir in os.listdir(path)]

    if shuffle_dirs:
        shuffle(dirs)

    for fn in get_fns(dirs):
        counter += 1

        if limit and counter > limit:
            return

        try:
            with open(fn) as f:
                data = json.load(f)

                yield data
        except:
            logging.error("Failed to load file %s ..." % fn)
