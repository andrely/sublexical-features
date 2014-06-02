import re

from nltk import ngrams

RE_ONE = re.compile('1', flags=re.MULTILINE)
RE_TWO = re.compile('2', flags=re.MULTILINE)
RE_THREE = re.compile('3', flags=re.MULTILINE)
RE_FOUR = re.compile('4', flags=re.MULTILINE)
RE_FIVE = re.compile('5', flags=re.MULTILINE)
RE_SIX = re.compile('6', flags=re.MULTILINE)
RE_SEVEN = re.compile('7', flags=re.MULTILINE)
RE_EIGHT = re.compile('8', flags=re.MULTILINE)
RE_NINE = re.compile('9', flags=re.MULTILINE)
RE_ZERO = re.compile('0', flags=re.MULTILINE)
RE_NOALPHA = re.compile('[^a-z]', flags=re.MULTILINE)
RE_WSPACE = re.compile('\s+', flags=re.MULTILINE)

def mahoney_clean(text_str):
    text_str = " " + text_str + " "
    text_str = text_str.lower()

    text_str = RE_ONE.sub(' one ', text_str)
    text_str = RE_TWO.sub(' two ', text_str)
    text_str = RE_THREE.sub(' three ', text_str)
    text_str = RE_FOUR.sub(' four ', text_str)
    text_str = RE_FIVE.sub(' five ', text_str)
    text_str = RE_SIX.sub(' six ', text_str)
    text_str = RE_SEVEN.sub(' seven ', text_str)
    text_str = RE_EIGHT.sub(' eight ', text_str)
    text_str = RE_NINE.sub(' nine ', text_str)
    text_str = RE_ZERO.sub(' zero ', text_str)

    text_str = RE_NOALPHA.sub(' ', text_str)
    text_str = RE_WSPACE.sub(' ', text_str)

    return text_str


def sublexicalize(text_str, order=3, join=True):
    text_str = re.sub(' ', '_', text_str)
    char_ngrams = ngrams(text_str, order)
    tokens = [''.join(token) for token in char_ngrams]

    if join:
        return ' '.join(tokens)
    else:
        return tokens


def make_preprocessor(order=None):
    if order:
        return lambda text_str: sublexicalize(mahoney_clean(text_str), order=order)
    else:
        return lambda text_str: mahoney_clean(text_str)
