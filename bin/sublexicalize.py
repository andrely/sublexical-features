from optparse import OptionParser
import os
import re
import sys

cur_path, _ = os.path.split(__file__)
sys.path.append(os.path.join(cur_path, '..', 'Experiments'))

from experiments.preprocessing import sublexicalize

BUF_SIZE = 8192

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-n", "--ngram-order", default=3)
    opts, args = parser.parse_args()

    order = int(opts.ngram_order)

    in_str = sys.stdin.read(BUF_SIZE)
    rest_str = ""

    while len(in_str) > 0:
        out_str = sublexicalize(rest_str + in_str.rstrip('\n'), order=order)
        rest_str = re.sub('_', ' ', out_str[-(order-1):])

        sys.stdout.write(out_str + " ")

        in_str = sys.stdin.read(BUF_SIZE)
