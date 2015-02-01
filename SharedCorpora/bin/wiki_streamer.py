from argparse import ArgumentParser
import logging
import sys
import bz2

from gensim.corpora.wikicorpus import extract_pages


def main():
    parser = ArgumentParser()
    parser.add_argument('-w', '--wiki-file')

    args = parser.parse_args()

    if args.wiki_file:
        f = bz2.BZ2File(args.wiki_file)
    else:
        f = sys.stdin

    texts = ((text, title, pageid) for title, text, pageid
             in extract_pages(f))

    for text, title, pageid in texts:
        text.replace('\n', ' ')
        sys.stdout.write(text.encode('utf-8'))
        sys.stdout.write('\n')

    f.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    main()