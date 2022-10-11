import warnings
import numpy as np
from sktime.classification.dictionary_based import MUSE

warnings.filterwarnings("ignore")

import glob
from timeit import default_timer as timer

from sklearn.metrics import f1_score, accuracy_score
from utils import ScaleData
import pickle
from mpi4py import MPI
import pandas as pd

dataset_dir_prefix = "./Datasets"

scaling_methods = ['minmax', 'maxabs', 'standard', 'robust', 'quantile', 'powert', 'normalize']
dimensions = ['timesteps', 'channels', 'both', 'all']

orig_comm = MPI.COMM_WORLD
orig_rank = orig_comm.Get_rank()

scaling_method = scaling_methods[orig_rank]

try:
    progress_list = pickle.load(open(f"muse_{scaling_method}_progress.pkl", 'rb'))
except FileNotFoundError:
    progress_list = []

for dimension in dimensions:
    global_time = 0
    for filename in sorted(glob.glob(F"{dataset_dir_prefix}/*.npz")):
        dataset = filename.split("/")[-1].split(".")[0]

        if dataset in ['InsectWingbeat', 'CharacterTrajectories', 'JapaneseVowels', 'SpokenArabicDigits',
                       'DuckDuckGeese', 'EigenWorms', 'FaceDetection', 'MotorImagery', 'PEMS-SF',
                       'Phoneme']:
            continue

        data = np.load(filename)
        orig_train_x, orig_test_x = data['train_x'].astype(np.float64), data['test_x'].astype(np.float64)
        train_y, test_y = data['train_y'], data['test_y']

        if scaling_method != "quantile":
            train_x, test_x = ScaleData(orig_train_x, orig_test_x, scaling_method, dimension, 0)

        global_start = timer()
        for seed in range(20):

            stats = []

            if scaling_method + "_" + dimension + "_" + dataset + "_" + str(seed) in progress_list:
                print(
                    f'Skipping Dataset : {dataset} - Seed {seed} - Method: {scaling_method} - Dimension: {dimension} because it has been calculated before.')
                continue

            np.random.seed(seed)

            if scaling_method == "quantile":
                train_x, test_x = ScaleData(orig_train_x, orig_test_x, scaling_method, dimension, seed)

            print(
                F"Rank {orig_rank}: Dataset : {dataset} - Seed {seed} - Method: {scaling_method} - Dimension: {dimension}",
                flush=True)

            muse = MUSE(window_inc=4, n_jobs=-1, random_state=seed)

            start = timer()
            muse.fit(train_x, train_y)
            end = timer()

            muse_fitting_time = end - start

            start = timer()
            predictions = muse.predict(test_x)
            end = timer()
            inference_time = end - start

            wf1 = f1_score(test_y, predictions, average='weighted')
            acc = accuracy_score(test_y, predictions)

            stats.append(
                [dataset, seed, muse_fitting_time, inference_time,
                 acc,
                 wf1])
            stats_df = pd.DataFrame.from_records(stats, columns=['Dataset', 'Seed', 'MUSE Fitting Time',
                                                                 'Inference time', 'Accuracy', 'Weighted F1'])

            stats_df.to_csv(f"muse_uea_metrics_{scaling_method}_{dimension}.csv", mode='a', header=False,
                            index=False)
            progress_list.append(scaling_method + "_" + dimension + "_" + dataset + "_" + str(seed))
            pickle.dump(progress_list, open(F"muse_{scaling_method}_progress.pkl", 'wb'))
