import io
import os

import numpy as np
import pandas as pd

FOLD_CSV = {
    'train': ('NLI_2013_Training_Data', 'index-training.csv'),
    'dev': ('NLI_2013_Development_Data', 'index-dev.csv')
}


def nli2013_dataset(dataset_path, fold=None):
    if not fold:
        fold = 'train'

    descr = pd.read_csv(os.path.join(dataset_path, FOLD_CSV[fold][0], FOLD_CSV[fold][1]), header=None,
                        names=['filename', 'prompt', 'l1', 'difficulty'])

    for i in range(len(descr)):
        row = descr.iloc[i]

        fn = row.filename

        with io.open(os.path.join(dataset_path, FOLD_CSV[fold][0], 'tokenized', fn), encoding='utf8') as f:
            text = f.read()

        yield {'text': text, 'l1': row.l1, 'prompt': row.prompt, 'difficulty': row.difficulty}


def nli2013_df(dataset_path, fold=None, limit=None):
    data = list(nli2013_dataset(dataset_path, fold=fold))

    if limit:
        data = np.random.choice(data, size=limit, replace=False)

    return pd.DataFrame(data)
