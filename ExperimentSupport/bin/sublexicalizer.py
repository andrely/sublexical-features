import logging
import sys
import codecs

from experiment_support.preprocessing import sublexicalize


sys.stdout=codecs.getwriter('utf-8')(sys.stdout)
sys.stdin=codecs.getreader('utf-8')(sys.stdin)


def main():
    for text in sys.stdin:
        text = sublexicalize(text, order=(3,6))

        sys.stdout.write(text)
        sys.stdout.write('\n')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    main()