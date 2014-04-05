import sys

from nltk import FreqDist


BUF_SIZE = 8192

if __name__ == '__main__':
    vocab = FreqDist()

    in_str = sys.stdin.read(BUF_SIZE)
    rest = ''

    read_count = 0

    while (rest + in_str).strip() != '':
        read_count += 1

        if read_count % 100 == 0:
            sys.stderr.write('.')
            sys.stderr.flush()

        tokens = (rest + in_str).split()
        rest = tokens.pop()

        if not tokens:
            vocab.update(rest)
            break
        else:
            vocab.update(tokens)

        in_str = sys.stdin.read(BUF_SIZE)

    print

    for i in [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]:
        if i > len(vocab.values()):
            break

        print "vocab size %7d - cutoff = %d" % (i, vocab.values()[i])
