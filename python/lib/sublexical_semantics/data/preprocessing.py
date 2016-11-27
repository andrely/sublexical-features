import re

import jieba
import nltk
import numpy as np
from pypinyin import pinyin
from spacy.en import English


def regex_tokenize(s, regex=r'\w+'):
    return re.findall(regex, s, re.MULTILINE | re.UNICODE)


def preprocess(texts):
    nlp = English()
    docs = nlp.pipe(texts)

    for doc in docs:
        for np in doc.noun_chunks:
            # Only keep adjectives and nouns, e.g. "good ideas"
            while len(np) > 1 and np[0].dep_ not in ('amod', 'compound'):
                np = np[1:]
            if len(np) > 1:
                # Merge the tokens, e.g. good_ideas
                np.merge(np.root.tag_, np.text, np.root.ent_type_)
            # Iterate over named entities
        for ent in doc.ents:
            if len(ent) > 1:
                # Merge them into single tokens
                ent.merge(ent.root.tag_, ent.text, ent.label_)

        sentences = []

        for sent in doc.sents:
            sentences.append([token.text for token in sent])

        yield sentences


def _word_tokenize(sent_str, lower_case=True):
    tokens = nltk.tokenize.TreebankWordTokenizer().tokenize(sent_str)

    if lower_case:
        tokens = [token.lower() for token in tokens]

    return tokens


class DataframeSentences():
    def __init__(self, df, limit=None, lower_case=True, cols=None, flatten=False):
        self.df = df
        self.limit = limit
        self.lower_case = lower_case
        self.cols = cols
        self.flatten = flatten

    def __iter__(self):
        for i in range(len(self.df)):
            if self.limit and i >= self.limit:
                break

            sents = []

            for col in self.cols:
                sents += [_word_tokenize(sent, lower_case=self.lower_case) for sent
                          in nltk.tokenize.PunktSentenceTokenizer().tokenize(self.df.iloc[i][col])]

            if self.flatten:
                yield [item for sublist in sents for item in sublist]
            else:
                for sent in sents:
                    yield sent

    def __len__(self):
        return len(self.df)


def zhang_ch_tokenization(text):
    return [''.join([item for sublist in pinyin(token) for item in sublist]) for token in jieba.cut(text)]


# from https://github.com/dennybritz/cnn-text-classification-tf
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()


def pad_sentences(sentences, padding_word="<PAD/>", max_length=None):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    if not max_length:
        sequence_length = max(len(x) for x in sentences)
    else:
        sequence_length = max_length

    padded_sentences = []

    for sentence in sentences:
        if len(sentence) > max_length:
            sentence = sentence[0:max_length]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)

    return padded_sentences


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
