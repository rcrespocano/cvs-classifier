# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DB_PATH = './data/cvs-db-20190318-export.csv'
TARGET = ['CVS_RASCH']
FEATURES = ['Z_USO_LC', 'Z_EDAD', 'Z_SEXO', 'Z_14DK', 'Z_14HIDRATACIÓN', 'Z_14DKt', 'Z_14LIMPIEZAYM', 'Z_15PASADO',
            'Z_15ANYOS', 'Z_15USO', 'Z_15DIASPORTE', 'Z_15REEMPLAZO', 'Z_22ORDENADOR', 'Z_23USO_ORDENADOR',
            'Z_24ATENCION_CONTINUA', 'Z_25PAUSAS_PAUTADAS', 'Z_26D_PAUSAS_PAUTADAS', 'Tipo_lente_ynosuarios_OR']


def ds_load():
    dataset = pd.read_csv(DB_PATH, na_values=' ')
    dataset = dataset.replace(np.nan, -1, regex=True)
    return dataset


def ds_train_test_datasets(dataset, train_size=0.80, verbose=False):
    train_x, test_x, train_y, test_y = train_test_split(dataset[FEATURES], dataset[TARGET], train_size=train_size)
    
    if verbose:
        print('Train_x Shape:', train_x.shape)
        print('Train_y Shape:', train_y.shape)
        print('Test_x Shape:', test_x.shape)
        print('Test_y Shape:', test_y.shape)
    
    return train_x, test_x, train_y, test_y
