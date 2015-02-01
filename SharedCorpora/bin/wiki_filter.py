from argparse import ArgumentParser
import logging
import sys
import codecs
import re

from gensim.corpora.wikicorpus import filter_wiki


PAT_ALPHABETIC = re.compile('(((?![\d])\w)+)', re.UNICODE)
PAT_NO_PUNCT = re.compile('(([\d\w])+)', re.UNICODE)
PAT_ALLOWED = re.compile('(([\.,\(\)\"&\\\/!\?:;\-\'\d\w])+)', re.UNICODE)
PAT_NORM_QUOTE = re.compile("''+")
PAT_NORM_NUM = re.compile('\d')


def tokenize(text, tok_type='all', norm_quote=True, norm_num=True, remove_underscore=True, lower=False):
    if lower:
        text = text.lower()

    if remove_underscore:
        text.replace('_', ' ')

    if norm_quote:
        text = PAT_NORM_QUOTE.sub('"', text)

    if tok_type == 'alpha':
        return [match.group() for match in PAT_ALPHABETIC.finditer(text)]
    elif tok_type == 'all' and norm_num:
        return [PAT_NORM_NUM.sub('#', match.group()) for match in PAT_ALLOWED.finditer(text)]
    elif tok_type == 'all':
        return [match.group() for match in PAT_ALLOWED.finditer(text)]
    elif tok_type == 'nopunct' and norm_num:
        return [PAT_NORM_NUM.sub('#', match.group()) for match in PAT_NO_PUNCT.finditer(text)]
    elif tok_type == 'nopunct':
        return [match.group() for match in PAT_NO_PUNCT.finditer(text)]
    else:
        raise NotImplementedError


def main():
    parser = ArgumentParser()
    parser.add_argument('-e', '--encoding')
    parser.add_argument('-t', '--tokenization', default='none')
    args = parser.parse_args()

    encoding = args.encoding
    tokenization = args.tokenization

    if encoding:
        sys.stdout = codecs.getwriter(encoding)(sys.stdout)
        sys.stdin = codecs.getreader(encoding)(sys.stdin)

    for text in sys.stdin:
        text = filter_wiki(text).strip()

        if tokenization != 'none':
            if tokenization == 'alpha':
                tokens = tokenize(text, tok_type='alpha', lower=False)
            elif tokenization == 'alpha-lower':
                tokens = tokenize(text, tok_type='alpha', lower=True)

            elif tokenization == 'all':
                tokens = tokenize(text, tok_type='all', lower=False, norm_num=True)
            elif tokenization == 'all-lower':
                tokens = tokenize(text, tok_type='all', lower=True, norm_num=True)
            elif tokenization == 'all-num':
                tokens = tokenize(text, tok_type='all', lower=False, norm_num=False)
            elif tokenization == 'all-lower-num':
                tokens = tokenize(text, tok_type='all', lower=True, norm_num=False)

            elif tokenization == 'nopunct':
                tokens = tokenize(text, tok_type='nopunct', lower=False, norm_num=True)
            elif tokenization == 'nopunct-lower':
                tokens = tokenize(text, tok_type='nopunct', lower=True, norm_num=True)
            elif tokenization == 'nopunct-num':
                tokens = tokenize(text, tok_type='nopunct', lower=False, norm_num=False)
            elif tokenization == 'nopunct-lower-num':
                tokens = tokenize(text, tok_type='nopunct', lower=True, norm_num=False)
            else:
                raise NotImplementedError

            text = ' '.join(tokens)

        if PAT_ALPHABETIC.match(text):
            sys.stdout.write(text + '\n')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    main()
