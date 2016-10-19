import json
import os
from glob import glob
from gzip import GzipFile

from pandas.core.frame import DataFrame


def amazon_reviews_iterator(dataset_path):
    fns = glob(os.path.join(dataset_path, '*.json.gz'))

    for fn in fns:
        with GzipFile(fn) as f:
            base_fn = os.path.basename(fn)

            for line in f:
                doc = json.loads(line.decode('utf-8'))
                doc['sourcefile'] = base_fn

                yield doc


def amazon_reviews_df(dataset_path):
    return DataFrame(data=amazon_reviews_iterator(dataset_path))
