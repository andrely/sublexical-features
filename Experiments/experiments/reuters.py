from collections import Counter

from sklearn.cross_validation import cross_val_score, KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from SharedCorpora.reuters import TopicSequence, corpus_path, ArticleSequence
from experiments.experiment_runner import TopicPipeline
from experiments.preprocessing import mahoney_clean


def corpus_statistics(corpus_path):
    article_count = 0
    topic_counts = Counter()

    for topics in TopicSequence(corpus_path):
        article_count += 1

        for topic in topics:
            topic_counts[topic] += 1

    return article_count, topic_counts


topics_by_freq = [u'earn', u'acq', u'money-fx', u'grain', u'crude', u'trade', u'interest', u'ship', u'wheat', u'corn',
                  u'oilseed', u'sugar', u'dlr', u'gnp', u'coffee', u'veg-oil', u'gold', u'nat-gas', u'money-supply',
                  u'livestock', u'soybean', u'bop', u'cpi', u'copper', u'carcass', u'reserves', u'jobs', u'cocoa',
                  u'rice', u'iron-steel', u'cotton', u'yen', u'alum', u'ipi', u'gas', u'meal-feed', u'rubber',
                  u'barley', u'zinc', u'palm-oil', u'pet-chem', u'silver', u'lead', u'rapeseed', u'sorghum', u'tin',
                  u'strategic-metal', u'wpi', u'fuel', u'hog', u'soy-meal', u'orange', u'soy-oil', u'heat', u'retail',
                  u'housing', u'sunseed', u'stg', u'lumber', u'tea', u'dmk', u'lei', u'oat', u'income', u'platinum',
                  u'nickel', u'groundnut', u'l-cattle', u'jet', u'rape-oil', u'sun-oil', u'coconut-oil', u'naphtha',
                  u'propane', u'inventories', u'coconut', u'potato', u'instal-debt', u'nzdlr', u'plywood', u'austdlr',
                  u'tapioca', u'pork-belly', u'palmkernel', u'lit', u'copra-cake', u'f-cattle', u'dfl', u'rand',
                  u'palladium', u'can', u'cotton-oil', u'saudriyal', u'sfr', u'cornglutenfeed', u'groundnut-oil',
                  u'linseed', u'castor-oil', u'fishmeal', u'sun-meal', u'lin-oil', u'wool', u'rye', u'nkr', u'cpu',
                  u'red-bean', u'rape-meal', u'bfr', u'corn-oil', u'lin-meal', u'ringgit', u'castorseed', u'dkr',
                  u'skr', u'citruspulp', u'hk', u'peseta', u'rupiah', u'cottonseed']

if __name__ == '__main__':
    topics = TopicSequence(corpus_path, include_topics=topics_by_freq[0:10])
    articles = ArticleSequence(corpus_path, preprocessor=mahoney_clean)

    model = TopicPipeline(CountVectorizer(max_features=1000,
                                          decode_error='ignore',
                                          strip_accents='unicode',
                                          preprocessor=mahoney_clean), MultinomialNB())

    scores = cross_val_score(model, articles, topics, verbose=2, scoring='f1',
                             cv=KFold(len(articles), n_folds=10, shuffle=True))

    print scores
