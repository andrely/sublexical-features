import logging
import sys
import time


def main():
    t_count = 0
    start = time.clock()
    prev = start

    for text in sys.stdin:
        t_count += len(text.split())

        sys.stdout.write(text)

        cur = time.clock()

        if cur - prev > 10:
            logging.info("Processed %d in %d seconds, %.0f t/s"
                         % (t_count, cur - start, t_count*1. / (cur - start)))

            prev = cur

    end = time.clock()
    logging.info("Processed %d finished in %d seconds, %.0f t/s"
                 % (t_count, end - start, t_count*1. / (end - start)))



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    main()