import pandas as pd
from rdkit import Chem
from mol2vec import features as feats
from gensim.models import word2vec
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import torch
from models import FCModel

from itertools import repeat
import time


def load_data(filepath):
    print('Loading data..')
    data = pd.read_csv(filepath)
    return data


def get_features(data, with_labels=True):
    print('Getting features..')
    if with_labels:
        all_mols = [(Chem.MolFromSmiles(row[1]['Smiles']), row[1]['Active']) for row in data[['Smiles', 'Active']].iterrows()]
    else:
        all_mols = [(Chem.MolFromSmiles(row[1]['Smiles']), None) for row in data[['Smiles']].iterrows()]
    mol_sentences = [(feats.mol2alt_sentence(x[0], 1), x[1]) for x in all_mols]

    vec_model = word2vec.Word2Vec.load('vecmodels/model_300dim.pkl')
    molvec_active = list()
    for x in mol_sentences:
        try:
            molvec_active.append((sum(vec_model.wv.get_vector(fragid) for fragid in x[0]), x[1]))
        except KeyError:
            molvec_active.append((np.zeros(300), x[1]))

    molvecs, molvec_labels = zip(*molvec_active)
    molvecs = torch.tensor(np.array(molvecs), dtype=torch.float32)
    molvec_labels = torch.tensor(np.array(molvec_labels), dtype=torch.float32) if with_labels else None

    return molvecs, molvec_labels


def fit_model(features, labels, model):
    print('Fitting model..')
    model.fit(features, labels, lr=1e-3, epochs=500)


def test_model(features, labels, model, data_type):
    print('Testing model..')
    pred = torch.tensor(model.predict(features), dtype=torch.float32)
    result = f1_score(labels, pred)
    print(f'F1 on {data_type} data:', result)
    return result


def create_submission_file(in_file_path, model):
    submission_data = load_data(in_file_path)
    sub_features, _ = get_features(submission_data, with_labels=False)
    preds = model.predict(sub_features)
    submission_data = submission_data.assign(Active=torch.tensor(preds, dtype=torch.int8))
    submission_data.to_csv(f'submissions/{time.time()}.csv', index=False)


def main():
    data = load_data('data/train.csv')
    features, labels = get_features(data)
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2)

    model = FCModel(features.shape[1], 1)
    fit_model(train_features, train_labels, model)

    f1_train = test_model(train_features, train_labels, model, 'train')
    f1_test = test_model(test_features, test_labels, model, 'test')

    print('Saving model..')
    torch.save(model.state_dict(), f'saved_models/train_{f1_train}_test_{f1_test}.mdl')

    print('=' * 20)
    create_submission_file('data/test.csv', model)


main()
