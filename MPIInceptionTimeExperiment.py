import warnings
import numpy as np

from utils import ScaleData

warnings.filterwarnings("ignore")

import glob
from timeit import default_timer as timer

from sklearn.metrics import f1_score, accuracy_score
from sktime_dl.deeplearning.inceptiontime._classifier import InceptionTimeClassifier
import pickle
from mpi4py import MPI
import pandas as pd
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

dataset_dir_prefix = "./Datasets"

scaling_methods = ['minmax', 'maxabs', 'standard', 'robust', 'quantile', 'powert', 'normalize']
dimensions = ['timesteps', 'channels', 'both', 'all']

orig_comm = MPI.COMM_WORLD
orig_rank = orig_comm.Get_rank()

scaling_method = scaling_methods[orig_rank]

try:
    progress_list = pickle.load(open(f"inct_{scaling_method}_progress.pkl", 'rb'))
except FileNotFoundError:
    progress_list = []

for dimension in dimensions:
    global_time = 0
    for filename in sorted(glob.glob(F"{dataset_dir_prefix}/*.npz")):
        dataset = filename.split("/")[-1].split(".")[0]

        if dataset in ['InsectWingbeat', 'CharacterTrajectories', 'JapaneseVowels', 'SpokenArabicDigits']:
            continue

        data = np.load(filename)
        orig_train_x, orig_test_x = data['train_x'].astype(np.float64), data['test_x'].astype(np.float64)
        orig_train_y, orig_test_y = data['train_y'], data['test_y']

        if scaling_method != "quantile":
            train_x, test_x = ScaleData(orig_train_x, orig_test_x, scaling_method, dimension, 0)

            ## InceptionTime requires time series length as second dimension, channels as third
            train_x = np.transpose(train_x, (0, 2, 1))
            test_x = np.transpose(test_x, (0, 2, 1))

        global_start = timer()
        for seed in range(20):

            stats = []

            if scaling_method + "_" + dimension + "_" + dataset + "_" + str(seed) in progress_list:
                print(
                    f'Skipping Dataset : {dataset} - Seed {seed} - Method: {scaling_method} - Dimension: {dimension} because it has been calculated before.')
                continue

            np.random.seed(seed)
            tf.compat.v1.set_random_seed(seed)

            if scaling_method == "quantile":
                train_x, test_x = ScaleData(orig_train_x, orig_test_x, scaling_method, dimension, seed)
                train_x = np.transpose(train_x, (0, 2, 1))
                test_x = np.transpose(test_x, (0, 2, 1))

            output_directory = f"./"

            print(
                F"Rank: {orig_rank} - Dataset : {dataset} - Seed {seed} - Method: {scaling_method} - Dimension: {dimension}",
                flush=True)
            if len(tf.config.list_physical_devices('GPU')) == 0:
                print("No GPU detected for InceptionTime")
                exit(-1)

            with tf.device('/device:GPU:0'):
                checkpoint_filepath = f'./{orig_rank}_best_inct.hdf5'
                model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_filepath,
                    save_weights_only=True,
                    monitor='loss',
                    save_best_only=True)
                es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-4, patience=150)
                inct = InceptionTimeClassifier(depth=2, nb_epochs=1500, verbose=False,
                                               callbacks=[model_checkpoint_callback, es_callback],
                                               random_state=seed,
                                               model_name=f"{orig_rank}_inct",
                                               model_save_directory=output_directory)

            start = timer()
            with tf.device('/device:GPU:0'):
                inct.fit(train_x, orig_train_y, input_checks=False)
            end = timer()

            inct_fitting_time = end - start

            last_epoch = len(inct.history.history['loss'])

            inct.model.load_weights(checkpoint_filepath)

            start = timer()
            y_pred = inct.predict_proba(test_x).argmax(axis=1)
            end = timer()
            inference_time = end - start

            wf1 = f1_score(orig_test_y, y_pred, average='weighted')
            acc = accuracy_score(orig_test_y, y_pred)

            stats.append(
                [dataset, seed, inct_fitting_time, last_epoch, inference_time,
                 acc,
                 wf1])
            stats_df = pd.DataFrame.from_records(stats, columns=['Dataset', 'Seed', 'Training Time', 'Last Epoch',
                                                                 'Inference time', 'Accuracy', 'Weighted F1'])

            stats_df.to_csv(f"inct_uea_metrics_{scaling_method}_{dimension}.csv", mode='a', header=False,
                            index=False)
            progress_list.append(scaling_method + "_" + dimension + "_" + dataset + "_" + str(seed))
            pickle.dump(progress_list, open(F"inct_{scaling_method}_progress.pkl", 'wb'))
