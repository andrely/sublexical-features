import bz2

from gensim.corpora.wikicorpus import filter_wiki, tokenize

from wiki_extractor import process_data


class WikiDump(object):
    def __init__(self, dump_fn, limit=None, tokenizer=None, stopwords=None):
        super(WikiDump, self).__init__()

        self.dump_fn = dump_fn
        self.limit = limit
        self.tokenizer = tokenizer or tokenize
        self.stopwords = stopwords or []

    def __iter__(self):
        with bz2.BZ2File(self.dump_fn) as f:
            page_count = 0
            for _, _, text in process_data(f):
                if self.limit and page_count > self.limit:
                    break

                yield [tok for tok in self.tokenizer(filter_wiki(text)) if tok not in self.stopwords]

                page_count += 1
