from argparse import ArgumentParser
import logging
import sys
import codecs

from experiment_support.preprocessing import sublexicalize, parse_ngram_order


def main():
    parser = ArgumentParser()
    parser.add_argument('-e', '--encoding')
    parser.add_argument('-o', '--order', default="3")
    args = parser.parse_args()

    encoding = args.encoding
    order = parse_ngram_order(args.order)

    if encoding:
        sys.stdout=codecs.getwriter(encoding)(sys.stdout)
        sys.stdin=codecs.getreader(encoding)(sys.stdin)

    for text in sys.stdin:
            text = sublexicalize(text, order=order)

            sys.stdout.write(text)
            sys.stdout.write('\n')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    main()