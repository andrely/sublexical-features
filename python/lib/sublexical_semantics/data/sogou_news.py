# coding=utf-8
import logging
import os
import re
from glob import glob

from pandas import DataFrame

EN_CATEGORIES = {u'传媒': 'media', '体育': 'sports', u'健康': 'health', u'公益': 'general',
                 u'军事': 'military', u'国内': 'domestic', u'国际': 'international', u'女人': 'woman',
                 u'女性': 'woman', u'娱乐': 'entertainment', u'媒体': 'media', u'房产': 'real estate',
                 u'招聘': 'jobs', u'教育': 'education', u'文化': 'culture', u'旅游': 'travel', u'时尚': 'fashion',
                 u'校园': 'campus', u'汽车': 'auto', u'社会': 'society', u'科技': 'technology', u'财经': 'finance',
                 u'IT': 'technology'}


def read_categories(fn):
    categories = []

    with open(fn, encoding='gb18030') as f:
        site = None
        cat = None
        for line in f:
            line = line.strip()

            if line == '':
                continue

            m = re.search('^\d+\.\s*(\w+)', line, re.UNICODE)

            if m:
                site = m.group(1)
                continue

            m = re.search('^http:', line)

            if m:
                categories.append((site, cat, line))
                continue

            cat = line

            # remove fullwidth colon
            if cat[-1] == u'\uFF1A':
                cat = cat[:-1]

    return categories


def read_docs(fns, limit=None):

    count = 0

    for fn in fns:
        logging.info('Reading file %s ...' % fn)

        with open(fn, encoding='gb18030') as f:
            doc = {}

            for line in f:
                line = line.strip()

                if line == '<doc>':
                    doc = {}
                elif line == '</doc>':
                    yield doc

                    count += 1

                    if count % 100000 == 0:
                        logging.info('Read %d documents ...' % count)

                    if limit and count >= limit:
                        return

                elif line.startswith('<url>'):
                    m = re.match('<url>(.*)</url>', line, re.UNICODE)
                    if m:
                        doc['url'] = m.group(1)
                elif line.startswith('<docno>'):
                    m = re.match('<docno>(.*)</docno>', line, re.UNICODE)
                    if m:
                        doc['docno'] = m.group(1)
                elif line.startswith('<content>'):
                    m = re.match('<content>(.*)</content>', line, re.UNICODE)
                    if m:
                        doc['content'] = m.group(1)
                elif line.startswith('<contenttitle>'):
                    m = re.match('<contenttitle>(.*)</contenttitle>', line, re.UNICODE)
                    if m:
                        doc['contenttitle'] = m.group(1)


def sogou_news_dataset(dataset_path, limit=None):
    categories = read_categories(os.path.join(dataset_path, 'categories_2012.txt'))
    fns = glob(os.path.join(dataset_path, 'Sogou*.mini.txt')) + glob(os.path.join(dataset_path, 'news*_xml.dat'))

    logging.info("Reading files %s ..." % ', '.join(fns))

    for doc in read_docs(fns, limit=limit):
        doc = add_category(doc, categories)

        yield doc


def add_category(doc, categories):
    for entry in categories:
        site, cat, url = entry

        if 'url' in doc and doc['url'].startswith(url):
            doc['cat'] = cat
            doc['cat_en'] = EN_CATEGORIES.get(cat)
            doc['site'] = site
            break

    return doc


def sogou_news_df(dataset_path, tokenizer=None, limit=None):
    def preproc(doc):
        if tokenizer:
            doc['content'] = ' '.join(tokenizer(doc['content']))
            doc['contenttitle'] = ' '.join(tokenizer(doc['contenttitle']))

        return doc

    return DataFrame(data=(preproc(doc) for doc in sogou_news_dataset(dataset_path, limit=limit)))
