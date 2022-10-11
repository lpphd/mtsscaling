import numpy as np
from sklearn.linear_model import RidgeClassifierCV
import warnings

warnings.filterwarnings("ignore")

import glob
from timeit import default_timer as timer

from sklearn.metrics import f1_score, accuracy_score
from sktime.transformations.panel.rocket import Rocket
from utils import ScaleData
import pickle

import pandas as pd

dataset_dir_prefix = "./Datasets"

scaling_methods = ['minmax', 'maxabs', 'standard', 'robust', 'quantile', 'powert', 'normalize']
dimensions = ['timesteps', 'channels', 'both', 'all']

try:
    progress_list = pickle.load(open("rocket_progress.pkl", 'rb'))
except FileNotFoundError:
    progress_list = []

for scaling_method in scaling_methods:
    for dimension in dimensions:
        global_time = 0
        for filename in sorted(glob.glob(F"{dataset_dir_prefix}/*.npz")):
            dataset = filename.split("/")[-1].split(".")[0]

            if dataset in ['InsectWingbeat', 'CharacterTrajectories', 'JapaneseVowels', 'SpokenArabicDigits']:
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

                print(F"Dataset : {dataset} - Seed {seed} - Method: {scaling_method} - Dimension: {dimension}",
                      flush=True)

                rocket = Rocket(normalise=False, random_state=seed, n_jobs=-1)

                start = timer()
                rocket.fit(train_x)
                end = timer()

                parameter_generation_time = end - start

                start = timer()
                X_training_transform = np.nan_to_num(rocket.transform(train_x), posinf=0, neginf=0)
                end = timer()

                kernel_application_time = end - start

                start = timer()

                classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
                classifier.fit(X_training_transform, train_y)

                end = timer()

                training_time = end - start

                start = timer()
                X_test_transform = np.nan_to_num(rocket.transform(test_x), posinf=0, neginf=0)
                predictions = classifier.predict(X_test_transform)
                end = timer()
                inference_time = end - start

                wf1 = f1_score(test_y, predictions, average='weighted')
                acc = accuracy_score(test_y, predictions)

                stats.append(
                    [dataset, seed, parameter_generation_time, kernel_application_time, training_time, inference_time,
                     acc,
                     wf1])
                stats_df = pd.DataFrame.from_records(stats, columns=['Dataset', 'Seed', 'Parameter Generation Time',
                                                                     'Train set transformation time', 'Training time',
                                                                     'Inference time', 'Accuracy', 'Weighted F1'])

                stats_df.to_csv(f"rocket_uea_metrics_{scaling_method}_{dimension}.csv", mode='a', header=False,
                                index=False)
                progress_list.append(scaling_method + "_" + dimension + "_" + dataset + "_" + str(seed))
                pickle.dump(progress_list, open("rocket_progress.pkl", 'wb'))
