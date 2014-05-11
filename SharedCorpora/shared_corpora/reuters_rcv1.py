from glob import glob
import logging
import os
from zipfile import ZipFile, zlib

from lxml import etree


SUBDIRS = ['100623_0655', '110104_0810']


def unpack_collection(path):
    archive_fns = glob(os.path.join(path, '*.zip'))

    for fn in archive_fns:
        subdir = os.path.splitext(os.path.basename(fn))[0]
        dest_path = os.path.join(path, subdir)

        logging.info("Extracting %s to %s" % (fn, dest_path))

        with ZipFile(fn) as z:
            try:
                z.extractall(dest_path)
            except zlib.error:
                logging.warn("Failed to inflate %s" % fn)


def corpus_fns(corpus_path):
    for dir in SUBDIRS:
        for item in glob(os.path.join(corpus_path, dir, '????????')):
            if os.path.isdir(item):
                for fn in glob(os.path.join(corpus_path, dir, item, '*.xml')):
                    yield fn


def read_code_map(f):
    map = {}
    rev_map = {}

    for line in f:
        line = line.strip()

        if line == '' or line[0] == ';':
            continue

        code, term = line.strip().split('\t')

        if map.has_key(code):
            map[code] += [term]
        else:
            map[code] = [term]

        if rev_map.has_key(term):
            rev_map[term] += [code]
        else:
            rev_map[term] = [code]

    return map, rev_map


def read_newsitem(fn):
    doc = etree.parse(fn)
    newsitem_elt = doc.getroot()

    if newsitem_elt.tag != 'newsitem':
        logging.warn("root tag %s in %s - skipping" % (newsitem_elt.tag, fn))
        return None

    newsitem = {}

    newsitem['id'] = newsitem_elt.attrib['itemid']
    newsitem['date'] = newsitem_elt.attrib['date']

    if newsitem_elt.find('title') != None:
        newsitem['title'] = newsitem_elt.find('title').text
    else:
        logging.warning("no title found in %s" % fn)

    if newsitem_elt.find('headline') != None:
        newsitem['headline'] = newsitem_elt.find('headline').text
    else:
        logging.warning("no headline found in %s" % fn)

    if newsitem_elt.find('byline') != None:
        newsitem['byline'] = newsitem_elt.find('byline').text
    else:
        logging.warning("no byline found in %s" % fn)

    if newsitem_elt.find('text') != None:
        newsitem['text'] = [p_elt.text for p_elt in newsitem_elt.find('text').findall('p')]
    else:
        logging.warning("no text found in %s" % fn)

    if newsitem_elt.find('metadata') != None:
        for codes_elt in newsitem_elt.find('metadata').findall('codes'):
            code_type = codes_elt.attrib['class'].split(':')[1]
            code_terms = [code_elt.attrib['code'] for code_elt in codes_elt.findall('code')]

            newsitem[code_type] = code_terms
    else:
        logging.warning("no metadata found in %s" % fn)

    return newsitem

