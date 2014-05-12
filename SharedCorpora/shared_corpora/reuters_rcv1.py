from collections import Counter
from glob import glob
import logging
import os
from zipfile import ZipFile, zlib, BadZipfile

from lxml import etree




# Reuters RCV1
# 803441 news items
# Average of 3.21 topics assigned to each article
# 103 different topics

# Topic distribution:
# CCAT 373856
# GCAT 234659
# MCAT 199850
# C15 149958
# ECAT 117433
# M14 84953
# C151 81796
# C152 72952
# GPOL 56840
# M13 52883
# C18 51423
# M11 48625
# M141 47625
# C181 43320
# E21 43106
# C17 41780
# C31 40446
# GDIP 37707
# C13 37362
# GSPO 35315
# GVIO 32586
# GCRIM 32191
# C24 32118
# M131 28145
# E212 27396
# E12 27055
# M132 26701
# M12 25994
# C21 25367
# C11 24297
# C1511 23200
# M143 21914
# E51 20689
# G15 19136
# C171 18287
# GJOB 17216
# E41 16876
# E211 15755
# C33 15310
# E512 12615
# M142 12115
# C12 11935
# C42 11857
# GVOTE 11530
# C172 11475
# C41 11341
# C411 10262
# GDEF 8834
# GDIS 8635
# E11 8561
# G154 8398
# C183 7403
# C14 7393
# C312 6639
# E13 6340
# GENV 6252
# C22 6114
# GHEA 6026
# C174 5869
# E131 5655
# GPRO 5494
# E71 5264
# C34 4827
# C182 4667
# C311 4297
# G158 4293
# GWEA 3868
# GENT 3798
# G151 3302
# E511 2923
# GREL 2846
# GODD 2793
# C173 2632
# C23 2621
# GSCI 2406
# G153 2359
# E31 2340
# E513 2290
# E121 2181
# E411 2135
# G155 2123
# G152 2103
# E14 2083
# C32 2081
# G157 2034
# C16 1918
# GWELF 1867
# E311 1699
# C331 1210
# E143 1205
# C313 1108
# E132 936
# GOBIT 844
# GTOUR 677
# E61 391
# E141 376
# GFAS 312
# G156 260
# E142 200
# E313 111
# E312 52
# G159 40
# GMIL 5

SUBDIRS = ['100623_0655', '110104_0810']

TOP5_TOPICS = ['CCAT', 'GCAT', 'MCAT', 'C15', 'ECAT']


def unpack_collection(path):
    archive_fns = glob(os.path.join(path, '*.zip'))

    for fn in archive_fns:
        subdir = os.path.splitext(os.path.basename(fn))[0]
        dest_path = os.path.join(path, subdir)

        logging.info("Extracting %s to %s" % (fn, dest_path))

        with ZipFile(fn) as z:
            try:
                z.extractall(dest_path)
            except (zlib.error, BadZipfile):
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
        pass
        # logging.warning("no byline found in %s" % fn)

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


def newsitem_gen(corpus_path):
    for fn in corpus_fns(corpus_path):
        yield read_newsitem(fn)


def topic_statistics(corpus_path):
    topic_counter = Counter()
    topic_avg = 0.0
    newsitem_count = 0

    for newsitem in newsitem_gen(corpus_path):
        try:
            topics = newsitem['topics']
            newsitem_count += 1

            topic_counter.update(topics)
            topic_avg += (len(topics) - topic_avg) / newsitem_count

            if newsitem_count % 10000 == 0:
                logging.info("Processed %d newsitems..." % newsitem_count)
        except KeyError:
            continue

    return newsitem_count, topic_avg, topic_counter
