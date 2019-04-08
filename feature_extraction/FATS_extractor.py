import sys
import os
import FATS
import numpy as np
import pickle as pkl
import sklearn
import time
import datetime

PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PATH_TO_PROJECT)


class ComputeFATS(object):

    def __init__(self, pkl_data_path):
        self.pkl_data_path = pkl_data_path


    def single_band_feature_keys(self):
        gal_feat_list = ['Amplitude', 'AndersonDarling', 'Autocor_length',
                         'Beyond1Std', 'CAR_sigma', 'CAR_mean', 'CAR_tau',
                         'Con', 'Eta_e', 'FluxPercentileRatioMid20',
                         'FluxPercentileRatioMid35',
                         'FluxPercentileRatioMid50',
                         'FluxPercentileRatioMid65',
                         'FluxPercentileRatioMid80',
                         'Gskew', 'LinearTrend',
                         'MaxSlope', 'Mean', 'Meanvariance', 'MedianAbsDev',
                         'MedianBRP', 'PairSlopeTrend', 'PercentAmplitude',
                         'PercentDifferenceFluxPercentile', 'PeriodLS',
                         'Period_fit', 'Psi_CS', 'Psi_eta', 'Q31', 'Rcs',
                         'Skew', 'SmallKurtosis', 'Std',
                         'StetsonK', 'VariabilityIndex']  # , 'StetsonK_AC']# 'SlottedA_length'

        harmonics_features = []
        for f in range(3):
            for index in range(4):
                harmonics_features.append("Freq" + str(f + 1) + "_harmonics_amplitude_" + str(index))
                harmonics_features.append("Freq" + str(f + 1) + "_harmonics_rel_phase_" + str(index))

        gal_feat_list += harmonics_features

        self.single_band_features_keys = gal_feat_list
        self.all_band_features = []

        for i in range(6):
            for f in gal_feat_list:
                self.all_band_features.append(f + "_" + str(i))

    def create_lightcurve_features(self):
        start_index = 0
        end_index = 0
        last_id = self.point_id[0]
        current_n = 0
        n_files_written = 0
        for i, ob_id in enumerate(self.point_id):
            if last_id != ob_id or i == (self.n_points_id - 1):
                current_n += 1
                #last_id = ob_id
                if i == (self.n_points_id - 1):
                    end_index = self.n_points_id
                else:
                    end_index = i
                lightcurve_frame = self.csv_data[start_index:end_index]
                features, short_sequence = self.compute_fats_features(lightcurve_frame)
                self.light_curve_features["object_id"].append(last_id)
                self.light_curve_features["features"].append(features)
                self.light_curve_features["short_sequence"].append(short_sequence)
                self.light_curve_features["lengths"].append(len(lightcurve_frame))
                n_files_written += 1
                #print(n_files_written)
                start_index = end_index
                last_id = ob_id
                #if n_files_written >= 100:
                #    break
        self.light_curve_features["object_id"] = np.array(self.light_curve_features["object_id"]).astype("S")
        self.light_curve_features["features"] = np.array(self.light_curve_features["features"])
        self.light_curve_features["short_sequence"] = np.array(self.light_curve_features["short_sequence"])
        self.light_curve_features["feature_list"] = np.array(self.all_band_features).astype("S")
        self.light_curve_features["lengths"] = np.array(self.light_curve_features["lengths"])
        hdf5_file = h5py.File(save_path, "w")
        dt = h5py.special_dtype(vlen=str)
        for data_key in self.light_curve_features.keys():
            if data_key in ["feature_list", "object_id"]:
                hdf5_file.create_dataset(data_key, data=self.light_curve_features[data_key], dtype=dt)
            else:
                hdf5_file.create_dataset(data_key, data=self.light_curve_features[data_key])


    def compute_fats_features(self, lightcurve_frame):

        def single_band_features(lc, errors, mjds):

            fats = FATS.FeatureSpace(Data=['magnitude', 'time', 'error'],
                                     featureList=self.single_band_features_keys,
                                     excludeList=['StetsonJ', 'StetsonL',
                                                  'Eta_color', 'Q31_color',
                                                  'Color'])
            data = np.array([lc, mjds, errors])
            features = fats.calculateFeature(data)
            return features  # , gal_feat_list

        band_list = np.unique(lightcurve_frame["passband"].values)
        lightcurve_features = []
        short_sequence = False
        for band in band_list:
            band_frame = lightcurve_frame[lightcurve_frame["passband"] == band]
            lc = band_frame["flux"].values
            errors = band_frame["flux_err"].values
            mjds = band_frame["mjd"].values
            if len(mjds) >= 4:
                features = single_band_features(lc=lc, errors=errors, mjds=mjds)
                features = features.result(method="array")
            else:
                features = np.zeros(shape=(58,))
                short_sequence = True
            lightcurve_features.append(features)

        lightcurve_features = np.concatenate(lightcurve_features)

        """combinations = list(itertools.combinations(band_list, 2))
        for pair in combinations:
            band_frame1 = lightcurve_frame[lightcurve_frame["passband"] == pair[0]]
            band_frame2 = lightcurve_frame[lightcurve_frame["passband"] == pair[1]]
        """

        return lightcurve_features, short_sequence

if __name__ == "__main__":

    index = sys.argv[1]
    csv_path = "../../training_data/partitions/test_set_part" + index + ".csv"
    save_path = "../../training_data/features/features_test_set" + index + ".hdf5"
    #save_path = "../../training_data/features/blablabla" + index + ".hdf5"


    feature_calculator = ComputeFATS(csv_path, save_path)
    start = time.time()
    feature_calculator.create_lightcurve_features()
print(csv_path, time.time()-start)