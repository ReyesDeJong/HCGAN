import pandas as pd
import numpy as np
import pickle

from tqdm import tqdm

import pytest


from modules.FATS.Feature import FeatureSpace

ts_path = "/Users/JonathanWihl/Desktop/FATS 2.0/CSSAllVarPhot.csv"
ts_save_path = "lcs.pkl"
ts_ids_save_path = "lcs_ids.pkl"
nrows = None

parallel = True
save = True 
save_directory = '.'
build_db = False

@pytest.fixture
def build_lcs(ts_path, ts_save_path, ts_ids_save_path, nrows=None):
    lcs_all = pd.read_csv(ts_path, nrows=nrows, header=None, names=["ID","MJD","mag","err","ra","dec"],sep=",")
    lcs_all = lcs_all.sort_values(['ID', 'MJD'])
    lcs_df = dict(tuple(lcs_all.groupby('ID')))
    lcs_keys = lcs_df.keys()
    lcs = []
    lcs_ids = []
    for _id in tqdm(lcs_keys):
        lc = lcs_df[_id]
        lc = lc[["mag", "MJD", "err"]]
        lcs.append(np.array(lc.values.T, dtype=np.float64))
        lcs_ids.append(_id)
    with open(ts_save_path,'wb') as f:
        pickle.dump(lcs, f)
    with open(ts_ids_save_path, 'wb') as f:
        pickle.dump(lcs_ids, f)

lc = build_lcs(ts_path, ts_save_path, ts_ids_save_path)

MostImportantFeatures = ['CAR_sigma','CAR_mean', 'Meanvariance', 'Mean', 'PercentDifferenceFluxPercentile',
                         'PercentAmplitude', 'Skew', 'Anderson-Darling test', 'Std',
                         'MedianAbsDev', 'Q31', 'Amplitude', 'PeriodLS']

def test_Amplitude():
    a = FeatureSpace(featureList=['Amplitude'])
    a.calculateFeature(lc)
