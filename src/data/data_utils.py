import pandas as pd
import os
import random


def read_train_test(type='train', full=False, size=1000000):
    '''Read training data'''
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    file_name = os.path.join(data_dir, 'processed', type + '.csv')

    dtypes = {
        'Semana': 'int32',
        'Agencia_ID': 'int32',
        'Canal_ID': 'int32',
        'Ruta_SAK': 'int32',
        'Cliente-ID': 'int32',
        'Producto_ID': 'int32',
        'Venta_hoy': 'float32',
        'Venta_uni_hoy': 'int32',
        'Dev_uni_proxima': 'int32',
        'Dev_proxima': 'float32',
        'Demanda_uni_equil': 'int32'
    }

    if not full:
        n = sum(1 for line in open(file_name)) - 1
        skip = sorted(random.sample(xrange(1, n + 1), n - size))

    # # get column names from test data
    # test1 = pd.read_csv(os.path.join(data_dir, 'processed', 'test.csv'),
    #                     nrows=5)
    # train1 = pd.read_csv(os.path.join(data_dir, 'processed', 'train.csv'),
    #                      nrows=5)

    # # get colum names to read from raw training data
    # read_cols = [x for x in train1.columns if x in test1.columns]
    # read_cols.append('Demanda_uni_equil')

    # read training data
    if full:
        df = pd.read_csv(file_name, dtype=dtypes) # usecols=read_cols,
    else:
        df = pd.read_csv(file_name, dtype=dtypes, # usecols=read_cols,
                         skiprows=skip)

    return df


def df_crossjoin(df1, df2, **kwargs):
    df1['_tmpkey'] = 1
    df2['_tmpkey'] = 1

    res = pd.merge(df1, df2, on='_tmpkey', **kwargs).drop('_tmpkey', axis=1)
    # res.index = pd.MultiIndex.from_product((df1.index, df2.index))

    df1.drop('_tmpkey', axis=1, inplace=True)
    df2.drop('_tmpkey', axis=1, inplace=True)

    return res
