import pandas as pd
import xgboost as xgb
import numpy as np
import os
import gc
import time

# from sklearn.metrics import mean_squared_error
# from math import sqrt
# from sklearn.grid_search import GridSearchCV


def xgb_model_predict():
    '''Run xgboost on processed data and make predictions'''
    proj_dir = os.path.join(os.path.dirname(__file__), '..', '..')

    #  to decrease memory usage:
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

    # read training data
    train = pd.read_csv(os.path.join(proj_dir, 'data', 'processed',
                                     'train.csv'), dtype=dtypes)

    train['demand_adj'] = np.log1p(train.Demanda_uni_equil)
    target = 'demand_adj'
    train.drop('Demanda_uni_equil', axis=1, inplace=True)

    predictors = [x for x in train.columns if x not in 'demand_adj']
    target = 'demand_adj'

    xgtrain = xgb.DMatrix(train[predictors].values, label=train[target].values)

    del train
    gc.collect()

    best_param = {
        'colsample_bytree': 0.9,
        'learning_rate': 0.5,
        'max_depth': 24,
        'subsample': 0.9,
        'objective': 'reg:linear',
        'n_round': 50
    }

    bst = xgb.train(params=best_param, dtrain=xgtrain, verbose_eval=True)

    test = pd.read_csv(os.path.join(proj_dir, 'data', 'processed', 'test.csv'),
                       dtype=dtypes)

    xgtest = xgb.DMatrix(test)

    preds = bst.predict(xgtest)
    preds = np.round(np.expm1(preds), 4)

    submission = test[['id']]
    submission = submission.assign(Demanda_uni_equil=preds)

    ts = time.strftime("%c").replace(' ', '-')

    submission_file = os.path.join(proj_dir, 'models', 'submissions',
                                   'xgb_' + ts + 'submission.csv')

    submission.to_csv(submission_file, index=False)


if __name__ == '__main__':
    xgb_model_predict()
