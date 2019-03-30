# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import savReaderWriter as pspp
import variables
from sklearn.model_selection import train_test_split


DB_PATH = './data/cvs-db-20190320-export.csv'
TARGET = 'CVS_RASCH'


def load(dataset_path, experiment_id=variables.EXP_N1, verbose=False):
    dataset = convert_sav_to_dataframe(dataset_path)
    dataset = pd.DataFrame(dataset, columns=variables.EXP_N1)

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

    return np.array(train_x), np.array(test_x), np.array(train_y, dtype=int), np.array(test_y, dtype=int)


def convert_sav_to_dataframe(dataset_path):
    # Read data
    with pspp.SavReader(dataset_path) as reader:
        # Read data
        _data = np.array(reader.all())

    # Read header
    with pspp.SavHeaderReader(dataset_path) as header:
        _columns = header.all()[0]
        _columns = [col.decode('UTF-8') for col in _columns]
        _metadata = np.array(_columns)

    # Dataframe
    return pd.DataFrame(_data, columns=_metadata)
