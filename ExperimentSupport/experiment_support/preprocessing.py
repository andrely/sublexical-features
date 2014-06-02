import re

from nltk import ngrams


def mahoney_clean(text_str):
    text_str = " " + text_str + " "
    text_str = text_str.lower()

    text_str = re.sub('1', ' one ', text_str, flags=re.MULTILINE)
    text_str = re.sub('2', ' two ', text_str, flags=re.MULTILINE)
    text_str = re.sub('3', ' three ', text_str, flags=re.MULTILINE)
    text_str = re.sub('4', ' four ', text_str, flags=re.MULTILINE)
    text_str = re.sub('5', ' five ', text_str, flags=re.MULTILINE)
    text_str = re.sub('6', ' six ', text_str, flags=re.MULTILINE)
    text_str = re.sub('7', ' seven ', text_str, flags=re.MULTILINE)
    text_str = re.sub('8', ' eight ', text_str, flags=re.MULTILINE)
    text_str = re.sub('9', ' nine ', text_str, flags=re.MULTILINE)
    text_str = re.sub('0', ' zero ', text_str, flags=re.MULTILINE)

    text_str = re.sub('[^a-z]', ' ', text_str, flags=re.MULTILINE)
    text_str = re.sub('\s+', ' ', text_str, flags=re.MULTILINE)

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
