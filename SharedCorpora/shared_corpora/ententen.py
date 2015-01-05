from gzip import GzipFile
import re


REMOVE_TAG_RE = re.compile('\s*<s>\s*(.*?)\s*</s>\s*')

def clean_line(line):
    m = REMOVE_TAG_RE.match(line)

    if len(m.groups()) == 1:
        return m.groups()[0]
    else:
        return ''


class EntentenCorpus(object):
    def __init__(self, corpus_fn, limit=None):
        super(EntentenCorpus, self).__init__()

        self.corpus_fn = corpus_fn

        self.limit = limit

    def __iter__(self):
        count = 0

        with GzipFile(self.corpus_fn) as f:
            for line in f:
                if self.limit and count > self.limit:
                    break

                tokens = clean_line(line).split()

                if tokens:
                    count += 1

                    yield tokens
