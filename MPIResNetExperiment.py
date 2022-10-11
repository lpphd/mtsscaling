import warnings
import numpy as np

from utils import ScaleData

warnings.filterwarnings("ignore")

import glob
from timeit import default_timer as timer

from sklearn.metrics import f1_score, accuracy_score
from sktime_dl.deeplearning.resnet._classifier import ResNetClassifier
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
    progress_list = pickle.load(open(f"resnet_{scaling_method}_progress.pkl", 'rb'))
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
        train_y, test_y = data['train_y'], data['test_y']

        if scaling_method != "quantile":
            train_x, test_x = ScaleData(orig_train_x, orig_test_x, scaling_method, dimension, 0)
            ## Resnet requires time series length as second dimension, channels as third
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
                print("No GPU detected for ResNet")
                exit(-1)

            with tf.device('/device:GPU:0'):
                checkpoint_filepath = f'./{orig_rank}_best_resnet.hdf5'
                model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_filepath,
                    save_weights_only=True,
                    monitor='loss',
                    save_best_only=True)
                es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-4, patience=150)
                resnet = ResNetClassifier(nb_epochs=1500, verbose=False,
                                          random_state=seed,
                                          callbacks=[model_checkpoint_callback, es_callback],
                                          model_name=f"{orig_rank}_resnet",
                                          model_save_directory=output_directory)

            start = timer()
            with tf.device('/device:GPU:0'):
                resnet.fit(train_x, train_y, input_checks=False)
            end = timer()

            resnet_fitting_time = end - start

            last_epoch = len(resnet.history.history['loss'])
            resnet.model.load_weights(checkpoint_filepath)

            start = timer()
            y_pred = resnet.predict_proba(test_x).argmax(axis=1)
            end = timer()
            inference_time = end - start

            wf1 = f1_score(test_y, y_pred, average='weighted')
            acc = accuracy_score(test_y, y_pred)

            stats.append(
                [dataset, seed, resnet_fitting_time, last_epoch, inference_time,
                 acc,
                 wf1])
            stats_df = pd.DataFrame.from_records(stats, columns=['Dataset', 'Seed', 'Training Time', 'LastEpoch',
                                                                 'Inference time', 'Accuracy', 'Weighted F1'])

            stats_df.to_csv(f"resnet_uea_metrics_{scaling_method}_{dimension}.csv", mode='a', header=False,
                            index=False)
            progress_list.append(scaling_method + "_" + dimension + "_" + dataset + "_" + str(seed))
            pickle.dump(progress_list, open(F"resnet_{scaling_method}_progress.pkl", 'wb'))
