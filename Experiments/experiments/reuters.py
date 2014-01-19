from collections import Counter

from corpora.reuters import corpus_articles


def corpus_statistics(corpus_path):
    article_count = 0
    topic_counts = Counter()

    for article in corpus_articles(corpus_path):
        article_count += 1

        for topic in article['topics']:
            topic_counts[topic] += 1

    return article_count, topic_counts


# def word_experiment(corpus_path, topic_cutoff=100):
    # n, topic_freqs = corpus_statistics(corpus_path)
    # vect = CountVectorizer((article))