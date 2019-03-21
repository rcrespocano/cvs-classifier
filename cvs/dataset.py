# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split


DB_PATH = './data/cvs-db-20190320-export.csv'
TARGET = 'CVS_RASCH'


def load(verbose=False):
    dataset = pd.read_csv(DB_PATH, na_values=' ')

    # Drop duplicated values
    dataset = dataset.drop_duplicates(subset=['ID'], keep='first')

    # Drop variables not useful
    dataset = dataset.drop('ID', axis=1)
    dataset = dataset.drop('Z_14TIPO_LENTE', axis=1)
    dataset = dataset.drop('DK_cat', axis=1)
    dataset = dataset.drop('Hidratacion_CAT', axis=1)
    dataset = dataset.drop('Punt_CVS_RASCH', axis=1)

    if verbose:
        print(dataset.head())

    return dataset


def train_test_datasets(dataset, train_size=0.80, verbose=False):
    X = dataset.drop(TARGET, axis=1)
    y = dataset[TARGET]

    train_x, test_x, train_y, test_y = train_test_split(X, y, train_size=train_size)

    if verbose:
        print('Train_x Shape:', train_x.shape)
        print('Train_y Shape:', train_y.shape)
        print('Test_x Shape:', test_x.shape)
        print('Test_y Shape:', test_y.shape)

    return train_x, test_x, train_y, test_y
