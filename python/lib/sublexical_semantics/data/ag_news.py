import os

import re

# Available at http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html


def ag_news_dataset(data_path, xml_fn='newsspace200.xml'):
    ag_fn = os.path.join(data_path, xml_fn)

    with open(ag_fn) as f:
        for line in f:
            line = line.strip()

            if line in ['<all_news>', '</all_news>', '<?xml version="1.0"?>']:
                continue

            source = re.search("<source>(.*)</source>", line).group(1)
            url = re.search("<url>(.*)</url>", line).group(1)
            title = re.search("<title>(.*)</title>", line).group(1)
            image = re.search("<image>(.*)</image>", line).group(1)
            category = re.search("<category>(.*)</category>", line).group(1)
            description = re.search("<description>(.*)</description>", line).group(1)
            rank = re.search("<rank>(.*)</rank>", line).group(1)
            pubdate = re.search("<pubdate>(.*)</pubdate>", line).group(1)

            yield {'source': source, 'url': url, 'title': title, 'image': image,
                   'category': category, 'description': description,
                   'rank': rank, 'pubdate': pubdate}
