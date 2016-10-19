import codecs
import os

from pandas import DataFrame

SUB_PATH = 'rt-polaritydata'
DATA_FNS = {'neg': 'rt-polarity.neg', 'pos': 'rt-polarity.pos'}


def sentence_polarity_dataset(data_path):
    if SUB_PATH:
        data_path = os.path.join(data_path, SUB_PATH)

    for polarity, fn in DATA_FNS.items():
        with codecs.open(os.path.join(data_path, fn), 'r', 'utf-8', errors='ignore') as f:
            for line in f:
                yield {'sentence': line.strip(), 'polarity': polarity}


def sentence_polarity_dataframe(data_path):
    return DataFrame(data=sentence_polarity_dataset(data_path), columns=['sentence', 'polarity'])
