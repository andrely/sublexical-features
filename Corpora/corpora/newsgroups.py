from glob import glob
import os

data_path = '/Users/stinky/Work/data'
corpus_path = os.path.join(data_path, '20_newsgroups')


def parse_fn(fn):
    with open(fn) as f:
        data = f.read()
        body = data[data.find('\n\n'):data.find('\n-- \n')]

    return body


def parse_articles(corpus_path):
    for path in glob(os.path.join(corpus_path, '*')):
        group = os.path.basename(path)

        for fn in glob(os.path.join(path, '*')):
            yield {'id': int(os.path.basename(fn)),
                   'group': group,
                   'body': parse_fn(fn)}

