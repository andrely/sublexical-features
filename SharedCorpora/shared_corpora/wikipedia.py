from gensim.corpora.textcorpus import TextCorpus

from gensim.corpora.wikicorpus import filter_wiki, tokenize

from wiki_extractor import process_data


class WikidumpCorpus(TextCorpus):
    def __init__(self, input=None, tokenizer=None, stopwords=None, limit=None):
        self.tokenizer = tokenizer or tokenize
        self.stopwords = stopwords or []
        self.limit = limit

        super(WikidumpCorpus, self).__init__(input)

    def get_texts(self):
        length = 0

        for _, _, text in process_data(self.input):
            length += 1

            yield [tok for tok in self.tokenizer(filter_wiki(text)) if tok not in self.stopwords]

            if self.limit and length >= self.limit:
                break

            self.length = length
