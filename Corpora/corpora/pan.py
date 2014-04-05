from collections import Sequence, Iterable
from glob import glob
import logging
import sqlite3
import os
import time

from lxml import etree
import psutil
from sklearn.feature_extraction.text import CountVectorizer

from experiments.experiment_runner import FilteredSequence


pan_ap_corpus_path = '/Users/stinky/Work/PAN2013/Author Profiling/pan_author_profiling_training_data'


class Timer(object):
    def __init__(self):
        self._start_time = None
        self._checkpoints = []
        self._end_time = None

    def start(self):
        self._start_time = time.time()

        return self

    def stop(self):
        self._end_time = time.time()

        return self.total()

    def checkpoint(self):
        cur_time = time.time()
        last_time = self._start_time

        if len(self._checkpoints) > 0:
            last_time = self._checkpoints[-1]

        self._checkpoints.append(cur_time)

        return cur_time - last_time

    def total(self):
        if self._start_time and self._end_time:
            return self._end_time - self._start_time
        else:
            return None


def pan_ap_files(corpus_path):
    return glob(os.path.join(corpus_path, 'en', '*.xml'))


def decode_fn_data(fn):
    base_fn = os.path.basename(fn)
    stripped_fn, _ = os.path.splitext(base_fn)
    author_id, lang, age, gender = stripped_fn.split('_')

    return author_id, lang, age, gender


def parse_fn(fn):
    author_id, lang, age, gender = decode_fn_data(fn)

    with open(fn) as f:
        doc = etree.HTML(f.read())

        author_nodes = doc.findall('.//author')

        if len(author_nodes) > 1:
            logging.info("Multiple author nodes in %s" % fn)

        author_node = author_nodes[0]

        stored_lang = author_node.attrib['lang']
        stored_gender = author_node.attrib['gender']
        stored_age = author_node.attrib['age_group']

        if stored_lang != lang:
            logging.warn("lang attribute '%s' differ from decoded fn lang '%s'" % (stored_lang, lang))

        if stored_gender != gender:
            logging.warn("gender attribute '%s' differ from decoded fn gender '%s'" % (stored_gender, gender))

        if stored_age != age:
            logging.warn("age attribute '%s' differ from decoded fn age '%s'" % (stored_age, age))

        conversations_nodes = doc.findall('.//conversations')

        if len(conversations_nodes) > 1:
            logging.info("Multiple conversations nodes in %s" % fn)

        conversations_node = conversations_nodes[0]

        conv_stored_count = int(conversations_node.attrib['count'])
        conv_count = 0

        for conversation in doc.findall('.//conversation'):
            conv_count += 1

            conv_id = conversation.attrib['id']
            content = conversation.text

            if content == None:
                content = ''

            yield (conv_id, author_id, lang, gender, age, unicode(content))

        if conv_count != conv_stored_count:
            logging.warn("Conversation count attr %d differ from actual count %d" % (conv_stored_count, conv_count))


def read_index(index_fn):
    logging.info("Reading index file %s" % index_fn)
    index = []

    with open(index_fn) as f:
        for line in f.readlines():
            author_id, conv_id = line.strip().split('\t')
            index.append((author_id.strip(), conv_id.strip()))

    return index


def create_index(index_fn, corpus_path):
    logging.info("Building index file %s" % index_fn)
    index = []

    with open(index_fn, 'w') as f:
        for fn in pan_ap_files(corpus_path):
            for conv_id, author_id, _, _, _, _ in parse_fn(fn):
                f.write("%s\t%s\n" % (author_id, conv_id))
                index.append((author_id, conv_id))

    logging.info("Completed index building")

    return index


def build_pan_ap_db(db_fn, ap_seq):
    with sqlite3.connect(db_fn) as c:
        c.execute("create table conversations (id integer primary key autoincrement, author_id text, conv_id text, lang text, gender text, age text, content text)")
        c.execute("create index author_index on conversations (author_id)")
        c.execute("create index conv_index on conversations (conv_id)")
        c.execute("create index author_conv_index on conversations (author_id, conv_id)")

        for vals in ap_seq:
            c.execute("insert into conversations (author_id, conv_id, lang, gender, age, content) values (?, ?, ?, ?, ?, ?)", vals)

    return db_fn


class PanAPSequence(Sequence):
    def __init__(self, corpus_path):
        self._corpus_path = corpus_path
        self._index_fn = os.path.join(corpus_path, 'en_index')

        if os.path.exists(self._index_fn):
            self._index = read_index(self._index_fn)
        else:
            self._index = create_index(self._index_fn, self._corpus_path)

        self._cached_parse = None

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self._get_index(i) for i in xrange(*index.indices(len(self)))]
        elif isinstance(index, Iterable):
            return [self._get_index(i) for i in index]
        elif isinstance(index, int):
            return self._get_index(index)
        else:
            raise TypeError

    def _get_index(self, index):
        index_author_id, index_conv_id = self._index[index]
        fns = glob(os.path.join(self._corpus_path, 'en', "%s_*.xml" % index_author_id))

        if len(fns) != 1:
            raise ValueError

        fn = fns[0]

        for conv_id, author_id, lang, gender, age, content in parse_fn(fn):
            if index_author_id != author_id:
                raise ValueError

            if index_conv_id == conv_id:
                return conv_id, author_id, lang, gender, age, content

        # no conversation found, should not happen
        raise ValueError

    def __len__(self):
        return len(self._index)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    logging.info("Running with PID %d" % os.getpid())

    timer = Timer().start()

    vect = CountVectorizer()
    seq = FilteredSequence(PanAPSequence(pan_ap_corpus_path), xrange(50000))

    mem_usage = psutil.Process(os.getpid()).get_memory_info()[0] / (1024 * 1024)
    logging.info("Mem usage %d MB, %.02f seconds"
                 % (mem_usage, timer.checkpoint()))

    vect.fit((content for conv_id, author_id, lang, gender, age, content
             in seq))

    mem_usage = psutil.Process(os.getpid()).get_memory_info()[0] / (1024 * 1024)
    logging.info("Mem usage %d MB, %.02f seconds"
                 % (mem_usage, timer.checkpoint()))

    print "%.02f" % timer.stop()
