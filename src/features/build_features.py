import pandas as pd
import numpy as np
import re
import os
import gc
import sys
# import feather

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
try:
    from data.data_utils import df_crossjoin
except:
    ImportError


def convert_to_grams(weight):
    '''convert all weight to grams'''
    w = re.sub(r'([A-z])', '', str(weight))
    kg = re.search(r'([K])', str(weight))

    if kg:
        try:
            grams = float(w) * 1000
        except:
            grams = np.nan
    else:
        try:
            grams = float(w)
        except:
            grams = np.nan

    return grams


def gen_prod_feats(prods):
    '''Generate misc features'''
    prods['brand'] = prods.NombreProducto.str.split(' ').str[-2]

    prods['weight'] = prods.NombreProducto.str.extract('(\d+[Kg])',
                                                       expand=False)

    prods['pieces'] = prods.NombreProducto.str.extract('(\\d+p\\b)',
                                                       expand=False)
    prods['pieces'] = prods.pieces.str.replace('p', '').astype('float')
    prods.pieces.fillna(1, inplace=True)
    prods['pieces'] = prods.pieces.astype('int')

    prods['grams'] = prods.weight.map(
        lambda x: convert_to_grams(x)).astype('float')

    # fill in missing values
    mdn_by_pieces = prods.groupby(['pieces'])['grams'].agg(
        'median').reset_index()
    mdn_by_pieces.columns = ['pieces', 'mdn_grams']

    prods = prods.merge(mdn_by_pieces, on='pieces', how='left')
    prods.grams.fillna(prods.mdn_grams, inplace=True)
    prods.drop('mdn_grams', axis=1, inplace=True)

    prods['grams_per_piece'] = np.round(prods.grams / prods.pieces, 3)

    # encode brands
    lbl_enc = LabelEncoder()
    prods['brand'] = lbl_enc.fit_transform(prods['brand'])

    prods.drop(['weight'], axis=1, inplace=True)

    return prods


def gen_prod_name_feats(prods):
    '''Generate additional features related to product names'''
    prods['short_name'] = prods.NombreProducto.str.extract('^(\D*)',
                                                           expand=False)
    prods['short_name_processed'] = (prods['short_name'].map(
        lambda x: " ".join(
            [i for i in x.lower().split() if i not in
                stopwords.words("spanish")])))

    stemmer = SnowballStemmer("spanish")

    prods['short_name_processed'] = prods['short_name_processed'].map(
        lambda x: " ".join([stemmer.stem(i) for i in x.lower().split()]))

    # encode short_name_processed
    lbl_enc = LabelEncoder()
    prods['short_name_processed'] = lbl_enc.fit_transform(
        prods.short_name_processed)

    prods.drop(['NombreProducto', 'short_name'], axis=1, inplace=True)

    return prods


def gen_lag_demand(data_dir):
    '''median demanda uni equil lagged 3 weeks'''
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

    train_file = os.path.join(data_dir, 'raw', 'train.csv')
    read_cols = ['Semana', 'Producto_ID', 'Cliente_ID', 'Demanda_uni_equil']

    train = pd.read_csv(train_file, usecols=read_cols, dtype=dtypes)

    agg = train.groupby(['Producto_ID', 'Cliente_ID', 'Semana'])[
        'Demanda_uni_equil'].agg('median').reset_index()

    del train
    gc.collect()

    pc = agg[['Producto_ID', 'Cliente_ID']].drop_duplicates()

    future_weeks = pd.DataFrame({'Semana': range(10, 12)})
    future = df_crossjoin(pc, future_weeks)
    future['Demanda_uni_equil'] = 0

    del pc, future_weeks
    gc.collect()

    full = pd.concat([agg, future], axis=0).reset_index()

    del agg, future
    gc.collect()

    full['Demand_uni_equil_lag3'] = full.groupby(
        ['Producto_ID', 'Cliente_ID'])['Demanda_uni_equil'].shift(3)

    full.drop('Demanda_uni_equil', axis=1, inplace=True)

    full = full[~np.isnan(full.Demand_uni_equil_lag3)]

    # full.to_hdf(os.path.join(data_dir, 'interim', 'features_demand_lag3.h5'),
    #             'lagd', mode='w', format='f', complevel=5, complib='zlib')
    # feather.write_dataframe(full, os.path.join(data_dir, 'interim',
    # 'features_demand_lag3.feather'))

    full.to_csv(os.path.join(data_dir, 'interim', 'features_demand_lag3.csv'),
                index=False)


def gen_lag_returns():
    '''3 week lagged returns'''
    pass


def gen_cliente_prod_freq():
    '''frequency of client-product pairs'''
    pass


def gen_cliente_count_by_town():
    '''cliente counts by town'''
    pass


def gen_cliente_count():
    '''cliente count by town_id'''
    pass


def build_features():
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

    # product attributes
    print 'Generating product attributes...'
    prods = pd.read_csv(os.path.join(data_dir, 'raw', 'producto_tabla.csv'))

    prods = gen_prod_feats(prods)
    prods = gen_prod_name_feats(prods)

    prods.to_csv(os.path.join(data_dir, 'interim',
                              'features_producto_tabla.csv'), index=False)

    # lagged demand
    print 'Generating lagged demand...'
    gen_lag_demand(data_dir)


if __name__ == '__main__':
    build_features()
