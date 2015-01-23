import re


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
RE_SPACE = re.compile(' ', flags=re.MULTILINE)


def normalize_whitespace(text_str):
    return RE_WSPACE.sub(' ', text_str)


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
    if not isinstance(text_str, unicode):
        text_str = text_str.decode('utf-8')


    if isinstance(order, int):
        order = range(order, order + 1)
    else:
        f, t = order
        order = range(f, t + 1)

    text_str = RE_SPACE.sub('_', text_str)

    ngrams = []

    # Fill in first "uneven" ngrams inefficiently

    n = order[-1]
    seqs = [zip(range(0, n-i), range(i, n)) for i in reversed(order)]

    for i in range(len(seqs)):
        seqs[i] = (len(seqs) - 1 - i)*[None] + seqs[i]

    indices = zip(*seqs)

    for i in indices:
        for j in i:
            if j:
                # noinspection PyUnresolvedReferences
                ngrams.append(text_str[j[0]:j[1]])

    # generate rest of ngrams fast

    n = len(text_str)

    for i in xrange(order[-1], n+1):
        for o in order:
            ngrams.append(text_str[i-o:i])

    if join:
        return ' '.join(ngrams)
    else:
        return ngrams


def make_preprocessor(order=None):
    if isinstance(order, (list, tuple)):
        return lambda text_str: [i for sl in [sublexicalize(mahoney_clean(text_str), order=o) for o in order] for i in sl]
    elif order:
        return lambda text_str: sublexicalize(mahoney_clean(text_str), order=order)
    else:
        return lambda text_str: mahoney_clean(text_str)


def clean_c35(text_str):
    return sublexicalize(mahoney_clean(text_str), order=(3, 5))


def clean_c45(text_str):
    return sublexicalize(mahoney_clean(text_str), order=(4, 5))


def clean_c36(text_str):
    return sublexicalize(mahoney_clean(text_str), order=(3, 6))
