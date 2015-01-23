import logging
import sys

from experiment_support.preprocessing import sublexicalize


def main():
    t_count = 0

    for text in sys.stdin:
        text = sublexicalize(text, order=(3,6))

        sys.stdout.write(' '.join(text))
        sys.stdout.write('\n')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    main()