import json
import logging
import os

from pandas import DataFrame
from pandas import concat


def flatten_votes(review_obj):
    if 'votes' in review_obj:
        review_obj.update({'votes_%s' % k: v for k, v in review_obj['votes'].items()})
        review_obj.pop('votes')

    return review_obj


def yelp_reviews(dataset_path):
    with open(os.path.join(dataset_path, 'yelp_academic_dataset_review.json')) as f:
        for line in f:
            obj = json.loads(line)
            flatten_votes(obj)

            yield obj


def yelp_reviews_df(dataset_path, limit=None, bulk_size=10000):
    dfs = []
    objs = []

    for i, review in enumerate(yelp_reviews(dataset_path)):
        if limit and i >= limit:
            break

        if i % 100000 == 0:
            logging.info("Processed %d reviews ..." % i)

        objs.append(review)

        if i % bulk_size == 0:
            df = DataFrame(data=objs)

            df.votes_useful = df.votes_useful.astype(int)
            df.votes_cool = df.votes_cool.astype(int)
            df.votes_funny = df.votes_funny.astype(int)

            dfs.append(df)

            objs = []

    if objs:
        df = DataFrame(data=objs)

        df.votes_useful = df.votes_useful.astype(int)
        df.votes_cool = df.votes_cool.astype(int)
        df.votes_funny = df.votes_funny.astype(int)

        dfs.append(df)

    return concat(dfs)
