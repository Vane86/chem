import pandas as pd
from rdkit import Chem
from mol2vec import features as feats
from gensim.models import word2vec
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt

import torch
from models import FCModel
from sklearn.neighbors import KNeighborsClassifier

from itertools import repeat
from functools import reduce
import time
import random

from utils import get_optimal_threshold

import warnings


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

N_FEATURES = 300
LIB_NAME = 'torch'

warnings.simplefilter('ignore')


def load_data(filepath):
    print('Loading data..')
    data = pd.read_csv(filepath)
    print()
    return data


def get_features(data, pca, pca_first=False, n_features=300, with_labels=True, lib='np'):
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
    molvecs = np.array(molvecs)
    molvec_labels = np.array(molvec_labels) if with_labels else None

    if pca_first:
        pca.fit(molvecs)
    molvecs = pca.transform(molvecs)
    print('Saved variance after PCA:', sum(pca.explained_variance_ratio_))

    if lib == 'torch':
        molvecs = torch.tensor(molvecs, dtype=torch.float32)
        molvec_labels = torch.tensor(molvec_labels, dtype=torch.float32) if with_labels else None
    print()

    return molvecs, molvec_labels


def fit_model(features, labels, model):
    print('Fitting model..')
    loss_history = list()
    model.fit(features, labels, lr=2.5e-4, epochs=2500, history=loss_history)
    # model.fit(features, labels)

    plt.plot(loss_history)
    plt.show()
    print()


def test_model(features, labels, model, data_type):
    print('Testing model..')
    pred_class = torch.tensor(model.predict(features), dtype=torch.float32)
    result = f1_score(labels, pred_class)
    precision, recall = precision_score(labels, pred_class), recall_score(labels, pred_class)
    pred_prob = torch.tensor(model.predict_prob(features), dtype=torch.float32)
    auc_roc = roc_auc_score(labels, pred_prob)
    print(f'AUC ROC on {data_type} data:', auc_roc)
    print(f'Precision on {data_type} data:', precision)
    print(f'Recall on {data_type} data:', recall)
    print(f'F1 on {data_type} data:', result)
    print()
    return result


def create_submission_file(in_file_path, pca, model):
    print('Creating submission file..')
    submission_data = load_data(in_file_path)
    sub_features, _ = get_features(submission_data, pca, n_features=N_FEATURES, with_labels=False, lib=LIB_NAME)
    preds = model.predict(sub_features)
    submission_data = submission_data.assign(Active=torch.tensor(preds, dtype=torch.int8))
    submission_data.to_csv(f'submissions/{time.time()}.csv', index=False)
    print()


def main():
    pca = PCA(n_components=N_FEATURES, random_state=123)

    data = load_data('data/train.csv')
    features, labels = get_features(data, pca, pca_first=True, n_features=N_FEATURES, lib=LIB_NAME)
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2)

    model = FCModel(features.shape[1], 1, p=0.7, threshold=0.39)
    fit_model(train_features, train_labels, model)
    # model.load_state_dict(torch.load('saved_models/train_0.99_test_0.225.mdl'))

    print('Optimal threshold:', get_optimal_threshold(test_labels, model.predict_prob(test_features)))

    f1_train = test_model(train_features, train_labels, model, 'train')
    f1_test = test_model(test_features, test_labels, model, 'test')

    if LIB_NAME == 'torch':
        print('Saving model..')
        torch.save(model.state_dict(), f'saved_models/train_{round(f1_train, 3)}_test_{round(f1_test, 3)}.mdl')

    print('=' * 20)
    create_submission_file('data/test.csv', pca, model)


main()
