import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
try:
    from model.xgb import xgb_model_predict
    from model.bag import ensemble_models
    from model.means_medians import medians_means_model_predict
except:
    ImportError


def make_final_predictions():
    '''make final predictions'''

    medians_means_model_predict()
    xgb_model_predict()
    os.system('pypy ~/projects/kaggle/grupo_bimbo/src/model/ftrl.py')

    ensemble_models()

if __name__ == '__main__':
    make_final_predictions()
