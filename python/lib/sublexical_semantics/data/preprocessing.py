import re
from spacy.en import English


def regex_tokenize(s, regex=r'\w+'):
    return re.findall(regex, s, re.MULTILINE | re.UNICODE)


def preprocess(texts):
    nlp = English()
    docs = nlp.pipe(texts)

    for doc in docs:
        for np in doc.noun_chunks:
            # Only keep adjectives and nouns, e.g. "good ideas"
            while len(np) > 1 and np[0].dep_ not in ('amod', 'compound'):
                np = np[1:]
            if len(np) > 1:
                # Merge the tokens, e.g. good_ideas
                np.merge(np.root.tag_, np.text, np.root.ent_type_)
            # Iterate over named entities
        for ent in doc.ents:
            if len(ent) > 1:
                # Merge them into single tokens
                ent.merge(ent.root.tag_, ent.text, ent.label_)

        sentences = []

        for sent in doc.sents:
            sentences.append([token.text for token in sent])

        yield sentences
