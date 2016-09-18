# coding: utf-8

import pandas as pd
import numpy as np
import os
import gc


def medians_means_model_predict():
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    train_file = os.path.join(data_dir, 'raw', 'train.csv')

    print 'Reading training data...'

    # read training data
    train = pd.read_csv(train_file, usecols=['Agencia_ID', 'Cliente_ID',
                                             'Producto_ID',
                                             'Demanda_uni_equil'])

    train = train.sort_values(['Agencia_ID', 'Cliente_ID', 'Producto_ID'])

    # overall median demand across all products, clients, agencies
    overall_mdn = train.Demanda_uni_equil.median()

    print 'Aggregating by Producto_ID, Agencia_ID...'

    # mean demand by Producto_ID, Agencia_ID
    prod_ag_mean = train.groupby(['Producto_ID', 'Agencia_ID'],
                                 as_index=False)['Demanda_uni_equil'].agg(
        lambda x: np.exp(np.mean(np.log(x + 1))) * 0.57941)

    prod_ag_mean.rename(columns={'Demanda_uni_equil': 'prod_ag_mean'},
                        inplace=True)

    gc.collect()

    print 'Aggregating by Producto_ID, Agencia_ID, Cliente_ID...'

    # mean demand by Producto_ID, Agencia_ID, Cliente_ID
    prod_ag_cli_mean = train.groupby(['Producto_ID', 'Agencia_ID',
                                      'Cliente_ID'],
                                     as_index=False)['Demanda_uni_equil'].agg(
        lambda x: np.exp(np.mean(np.log(x + 1))) - 0.91)

    prod_ag_cli_mean.rename(columns={'Demanda_uni_equil': 'prod_ag_cli_mean'},
                            inplace=True)

    del train
    gc.collect()

    print 'Creating Submission...'

    # read in test file
    test_file = os.path.join(data_dir, 'raw', 'test.csv')
    test = pd.read_csv(test_file)

    # create submission df by merging the means and median to test df
    submission = pd.merge(test, prod_ag_mean, on=['Producto_ID', 'Agencia_ID'],
                          how='left')

    submission = pd.merge(submission, prod_ag_cli_mean, on=['Producto_ID',
                          'Agencia_ID', 'Cliente_ID'], how='left')

    # take prod_ag_cli_mean where possible
    submission['Demanda_uni_equil'] = submission.prod_ag_cli_mean

    # next, take prod_ag_mean where prod_ag_cli_mean was NaN
    submission.Demanda_uni_equil.fillna(submission.prod_ag_mean, inplace=True)

    # finally, use the median where others were NaN
    submission.Demanda_uni_equil.fillna(overall_mdn, inplace=True)

    submission_dir = os.path.join(data_dir, '..', 'models', 'submissions',
                                  'means_medians.csv')

    submission[['id', 'Demanda_uni_equil']].to_csv(
        submission_dir, index=False)


if __name__ == '__main__':
    medians_means_model_predict()
