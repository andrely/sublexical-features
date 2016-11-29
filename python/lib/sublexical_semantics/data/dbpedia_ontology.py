import os

import bs4
import pandas
import requests

SUBCLASSES_QUERY = """
SELECT ?class
 WHERE {
     ?class rdfs:subClassOf <%s> .
 }
"""

ONTOLOGY_CLASS_COUNT_QUERY = """
select COUNT DISTINCT ?type {
   ?type a owl:Class .
}
"""

ONTOLOGY_CLASS_QUERY = """
select ?type {
   ?type a owl:Class .
}
"""

ONTOLOGY_CLASS_RESOURCE_COUNT_QUERY = """
PREFIX ontology:<http://dbpedia.org/ontology/>
SELECT COUNT DISTINCT(?person)
WHERE {
?person rdf:type ontology:%s .
}
"""

TITLE_AND_ABSTRACT_QUERY = """
PREFIX ontology:<http://dbpedia.org/ontology/>
SELECT DISTINCT(?person) ?title ?abstract
WHERE {
?person rdf:type ontology:%s .
?person <http://dbpedia.org/ontology/abstract> ?abstract .
?person foaf:name ?title .
FILTER langMatches(lang(?abstract),'en')
}
ORDER BY %s(?person)
OFFSET %s
"""

DISTINCT_CLASSES = ['SportsTeam', 'SocietalEvent', 'MeanOfTransportation', 'Plant', 'EducationalInstitution',
                    'OfficeHolder', 'WrittenWork', 'NaturalPlace', 'Building', 'Company',
                    'Infrastructure', 'Film', 'SoccerPlayer', 'Artist']


def dbpedia_ontology_dataset(data_path):
    dataframes = []

    for category in DISTINCT_CLASSES:
        df = pandas.read_csv(os.path.join(data_path, '%s.csv' % category.lower()),
                             encoding='utf8', index_col=False)
        df.dropna(inplace=True)
        df['category'] = category
        df.drop_duplicates(subset=['abstract'], inplace=True)

        dataframes.append(df)

    full_df = pandas.concat(dataframes)
    full_df.reset_index(drop=True, inplace=True)

    return full_df


def download_dbpedia_ontology_dataset(dataset_path):
    for category in DISTINCT_CLASSES:
        df = get_resource_data_for_class(category)

        df.to_csv(os.path.join(dataset_path, '%s.csv' % category.lower()),
                  index=False, encoding='utf8')


def get_resource_data_for_class(class_name):
    data = []

    for offset in [0, 10000, 20000, 30000]:
        resp = requests.get('http://dbpedia.org/sparql',
                            {'query': TITLE_AND_ABSTRACT_QUERY % (class_name, 'ASC', offset)})
        doc = bs4.BeautifulSoup(resp.content)
        results = [result for result in doc.findAll('result')]
        inner_data = [(r.find('binding', {'name': 'title'}).text, r.find('binding', {'name': 'abstract'}).text)
                      for r in results]

        data += inner_data

    for offset in [0, 10000, 20000, 30000]:
        resp = requests.get('http://dbpedia.org/sparql',
                            {'query': TITLE_AND_ABSTRACT_QUERY % (class_name, 'DESC', offset)})
        doc = bs4.BeautifulSoup(resp.content)
        results = [result for result in doc.findAll('result')]
        inner_data = [(r.find('binding', {'name': 'title'}).text, r.find('binding', {'name': 'abstract'}).text)
                      for r in results]

        data += inner_data

    return pandas.DataFrame(data, columns=['title', 'abstract']).drop_duplicates()


def get_count_for_class(class_name):
    resp = requests.get('http://dbpedia.org/sparql',
                        {'query': ONTOLOGY_CLASS_RESOURCE_COUNT_QUERY % class_name})
    doc = bs4.BeautifulSoup(resp.content)
    return int(doc.find('literal').text)


def get_classes():
    resp = requests.get('http://dbpedia.org/sparql', {'query': ONTOLOGY_CLASS_QUERY})
    doc = bs4.BeautifulSoup(resp.content)

    return set([uri.text.split('/')[-1] for uri in doc.findAll('uri') if 'ontology' in uri.text])


def get_classes_and_counts():
    return [(n, get_count_for_class(n)) for n in get_classes()]


def ontology_tree_inner(root):
    leaves = root.findAll('li', recursive=False)

    result = {}

    for leaf in leaves:
        c = leaf.find('a', href=True, recursive=False)

        if c:
            n = c['href']

            count = get_count_for_class(n)
            b = leaf.find('ul', recursive=False)
            result[n] = ontology_tree(b)

    return result


def ontology_tree():
    resp = requests.get('http://mappings.dbpedia.org/server/ontology/classes/')
    doc = bs4.BeautifulSoup(resp.content)

    return ontology_tree_inner(doc.find('ul'))
