import os
import sys

PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PATH_TO_PROJECT)
from xgboost import XGBClassifier
from parameters import param_keys

"""
RF wrapper, first designed for projections
"""


class RandomForest(object):

  def __init__(self, params_to_update=None):
    super().__init__(params_to_update)

    def get_clf(self):
      return XGBClassifier(**self.params[param_keys.CLF_PARAMS])

    def get_clf_default_params(self):
      clf_default_params = {'n_jobs': -1, 'max_depth': 9, 'n_estimators': 200,
                            'learning_rate': 0.1, 'subsample': 1.0,
                            'colsample_bytree': 0.8, 'gamma': 0.5,
                            'min_child_weight': 5,
                            'objective': 'multi:softprob'}
      return clf_default_params
