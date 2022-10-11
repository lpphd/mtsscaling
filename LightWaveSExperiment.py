import glob
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

from utils import ScaleData

warnings.filterwarnings("ignore")

from lightwavesl1l2_functions import _generate_first_phase_kernels, _apply_2layer_kernels
from lightwaves_utils import anova_feature_selection, mrmr_feature_selection, get_fixed_candidate_kernels, \
    ckd_to_kernels, transform, get_ckd_matrix_with_features
from sklearn.linear_model import RidgeClassifierCV

from mpi4py import MPI

## In the following section we set up variables for LightWaveS

## 4 features per scattering level, according to the L1L2 default version
features_number = 4
features_number *= 2

## Max dilation of kernels
MAX_DILATION = 32
# Final number of features
FINAL_NUM_FEAT = 500
## Pool of features selected on each node, before the final selection. Ensures correct distributed behavior and gives the feature pool for the mrmr method to work on
PRE_FINAL_FEAT_NUM = 3 * FINAL_NUM_FEAT
## Sample size of datasets, select up to this number of samples for larger datasets
SAMPLE_SIZE = 1500
## Numbe of base kernels
N_KERNELS = 84
VERBOSE = 0
## Depending on number of channels and distribute nodes, the MPI ranks may change during execution
orig_comm = MPI.COMM_WORLD
orig_rank = orig_comm.Get_rank()
orig_n_nodes = orig_comm.Get_size()

# Dilations array
dilations = np.arange(0, np.log2(MAX_DILATION) + 1).astype(np.int32)
n_dilations = dilations.size

##LightWaveS has been set up, the rest of the experiment is set up below

dataset_dir_prefix = "./Datasets"

## Set up a pickle file with the experiment progress to avoid repetition in case of interruption
try:
    progress_list = pickle.load(open("lightwaves_progress.pkl", 'rb'))
except FileNotFoundError:
    progress_list = []

scaling_methods = ['minmax', 'maxabs', 'standard', 'robust', 'quantile', 'powert', 'normalize']
dimensions = ['timesteps', 'channels', 'both', 'all']
for scaling_method in scaling_methods:
    for dimension in dimensions:
        for fileidx, filename in enumerate(sorted(glob.glob(F"{dataset_dir_prefix}/*.npz"))):

            orig_comm.Barrier()

            dataset = filename.split("/")[-1].split(".")[0]
            data_npz = np.load(filename)

            if dataset in ['InsectWingbeat', 'CharacterTrajectories', 'JapaneseVowels', 'SpokenArabicDigits']:
                continue

            ## Scaling methods are unaffected by random seed, apart from quantile
            if scaling_method != "quantile":
                orig_train_x, orig_test_x = ScaleData(data_npz['train_x'].astype(np.float64),
                                                      data_npz['test_x'].astype(np.float64), scaling_method, dimension,
                                                      0)
            orig_train_y, orig_test_y = data_npz['train_y'], data_npz['test_y']

            for seed in range(20):
                orig_comm.Barrier()
                np.random.seed(seed)

                if scaling_method + "_" + dimension + "_" + dataset + "_" + str(seed) in progress_list:
                    print(
                        f'Skipping Dataset : {dataset} - Seed {seed} - Method: {scaling_method} - Dimension: {dimension} because it has been calculated before.')
                    continue

                np.random.seed(seed)

                if orig_rank == 0:
                    print(F"Dataset : {dataset} - Seed {seed} - Method: {scaling_method} - Dimension: {dimension}",
                          flush=True)

                if scaling_method == "quantile":
                    orig_train_x, orig_test_x = ScaleData(data_npz['train_x'].astype(np.float64),
                                                          data_npz['test_x'].astype(np.float64), scaling_method,
                                                          dimension,
                                                          seed)

                total_num_channels = orig_train_x.shape[1]

                if total_num_channels < orig_n_nodes:
                    if orig_rank == 0:
                        if VERBOSE:
                            print("Number of channels is smaller than number of nodes, reducing COMM to subset.")
                    comm = orig_comm.Create_group(orig_comm.group.Incl(np.arange(total_num_channels).tolist()))
                else:
                    comm = orig_comm

                ## Split channels across MPI nodes
                channel_distribution = np.array_split(np.arange(total_num_channels), orig_n_nodes)

                ## Get channels of this node
                my_channels = channel_distribution[orig_rank]

                if orig_rank < total_num_channels:

                    rank = comm.Get_rank()
                    n_nodes = comm.Get_size()

                    train_x, train_y = orig_train_x[:, my_channels, :], orig_train_y

                    train_shape = train_x.shape
                    train_samples = train_shape[0]
                    num_channels = train_shape[1]
                    n_timepoints = train_shape[2]
                    num_classes = len(np.unique(train_y))

                    candidate_kernels = get_fixed_candidate_kernels()
                    n_candidate_kernels = len(candidate_kernels)

                    train_x = train_x.astype(np.float32)

                    if rank == 0:
                        if VERBOSE:
                            print(fileidx, dataset, seed)
                            print(candidate_kernels.shape[0] * n_dilations * total_num_channels)
                    first_phase_kernels = _generate_first_phase_kernels(num_channels, candidate_kernels,
                                                                        dilations, seed)

                    ## Get samples if training size is larger than limit
                    if train_samples > SAMPLE_SIZE:
                        np.random.seed(seed)
                        sample_idces = np.random.choice(train_samples, size=SAMPLE_SIZE, replace=False)
                        train_samples = SAMPLE_SIZE
                    else:
                        sample_idces = slice(None)

                    ## Transform train set
                    transform_features = _apply_2layer_kernels(train_x[sample_idces, ...], first_phase_kernels)

                    ## Select best features with ANOVA method
                    sel_feat_idces, sel_feat_scores = anova_feature_selection(
                        transform_features.reshape((transform_features.shape[0], -1)), train_y[sample_idces],
                        PRE_FINAL_FEAT_NUM)

                    ##Send feature scores to main node for comparison
                    ##First send number of features to main node
                    feat_count = np.array(sel_feat_idces.size).reshape((1, 1))
                    feat_count_recvbuf = None
                    if rank == 0:
                        feat_count_recvbuf = np.empty([n_nodes], dtype='int')
                    comm.Gather(feat_count, feat_count_recvbuf, root=0)

                    ## Then send actual scores to main node
                    displ = None
                    feat_scores_recvbuf = None
                    counts = None
                    feat_score_sendbuf = sel_feat_scores.flatten()
                    if rank == 0:
                        displ = np.hstack((0, feat_count_recvbuf.flatten())).cumsum()[:-1]
                        feat_scores_recvbuf = np.empty((feat_count_recvbuf.sum()), dtype=np.float32)
                        counts = feat_count_recvbuf

                    comm.Gatherv(feat_score_sendbuf, [feat_scores_recvbuf, counts, displ, MPI.FLOAT], root=0)

                    ## Main node sorts scores and sends back to each node how many (if any) of its top features to send
                    if rank == 0:
                        score_src_idces = []
                        for i in range(n_nodes):
                            score_src_idces.extend([i] * feat_count_recvbuf[i])
                        score_src_idces = np.array(score_src_idces)

                        top_score_src_count = np.bincount(
                            score_src_idces[np.argsort(feat_scores_recvbuf.flatten())[::-1]][
                            :PRE_FINAL_FEAT_NUM], minlength=n_nodes).astype(np.int32)

                    else:
                        top_score_src_count = np.empty(n_nodes, dtype=np.int32)

                    comm.Bcast(top_score_src_count, root=0)

                    ## On each node, select top features (if any)
                    sel_feat_idces = np.sort(
                        sel_feat_idces[np.argsort(sel_feat_scores)[::-1]][:top_score_src_count[rank]])

                    if (top_score_src_count == 0).any():
                        if orig_rank == 0 and VERBOSE == 1:
                            print("Some nodes have 0 CKD selected, reducing COMM to subset.")

                        new_comm = comm.Create_group(comm.group.Incl(np.where(top_score_src_count != 0)[0].tolist()))
                    else:
                        new_comm = comm

                    if top_score_src_count[rank] > 0:
                        rank = new_comm.Get_rank()
                        n_nodes = new_comm.Get_size()

                        ## Transform node feature indices to final format of channel-kernel-dilation-feature
                        ckdf = get_ckd_matrix_with_features(sel_feat_idces, num_channels, n_candidate_kernels,
                                                            n_dilations,
                                                            features_number)
                        ckdf[:, 0] = my_channels[ckdf[:, 0]]

                        ##Send kernel matrices to main node for second comparison
                        displ = None
                        ckdf_recvbuf = None
                        counts = None
                        feat_sendbuf = ckdf.flatten()
                        if rank == 0:
                            displ = np.hstack((0, top_score_src_count[top_score_src_count != 0].flatten())).cumsum()[
                                    :-1] * 4
                            ckdf_recvbuf = np.empty((4 * top_score_src_count.sum()), dtype=np.int32)
                            counts = top_score_src_count[top_score_src_count != 0] * 4

                        new_comm.Gatherv(feat_sendbuf, [ckdf_recvbuf, counts, displ, MPI.INT], root=0)

                        if rank == 0:
                            ckdf_recvbuf = ckdf_recvbuf.reshape((-1, 4))
                            test_y = orig_test_y
                            full_train_x, full_test_x = orig_train_x.astype(np.float32), orig_test_x.astype(np.float32)

                            ## On main node, keep unique kernels (some kernels may give more than 1 feature)
                            unique_ckdf_recvbuf = np.unique(ckdf_recvbuf[:, :-1], axis=0)

                            ## Create kernel matrix and feature mask
                            cand_kernels = ckd_to_kernels(unique_ckdf_recvbuf, candidate_kernels, dilations)
                            feat_mask = np.zeros((unique_ckdf_recvbuf.shape[0], features_number), dtype=bool)
                            sel_feat_per_k = list(pd.DataFrame(ckdf_recvbuf).groupby([0, 1, 2])[3].apply(list))
                            for i in range(feat_mask.shape[0]):
                                feat_mask[i, sel_feat_per_k[i]] = True

                            ## Transform train set with list of received kernels
                            cand_feats = _apply_2layer_kernels(full_train_x[sample_idces, :, :], cand_kernels)[:,
                                         feat_mask]

                            ## Select best features with mrmr method
                            global_sel_feats_p2_idces, _, _ = \
                                mrmr_feature_selection(cand_feats,
                                                       train_y[sample_idces],
                                                       FINAL_NUM_FEAT)

                            ## Keep best features from the previously received kernel set, generate final kernel matrix and feature mask
                            ckdf_recvbuf = ckdf_recvbuf[global_sel_feats_p2_idces, :]
                            kernel_matrix_final = np.unique(ckdf_recvbuf[:, :-1], axis=0)
                            feat_mask = np.zeros((kernel_matrix_final.shape[0], features_number), dtype=bool)
                            sel_feat_per_k = list(pd.DataFrame(ckdf_recvbuf).groupby([0, 1, 2])[3].apply(list))
                            for i in range(feat_mask.shape[0]):
                                feat_mask[i, sel_feat_per_k[i]] = True

                            ## Transform train and test set for final linear classification
                            train_tr = transform(full_train_x, kernel_matrix_final, feat_mask, candidate_kernels,
                                                 dilations)
                            test_tr = transform(full_test_x, kernel_matrix_final, feat_mask, candidate_kernels,
                                                dilations)

                            final_classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
                            final_classifier.fit(train_tr, train_y)
                            y_hat = final_classifier.predict(test_tr)
                            f1_sc = f1_score(test_y, y_hat, average='weighted')
                            acc = accuracy_score(test_y, y_hat)

                            stats_df = pd.DataFrame.from_records([[dataset, seed,
                                                                   acc,
                                                                   f1_sc]],
                                                                 columns=['Dataset', 'Seed', 'Accuracy', 'Weighted F1'])
                            stats_df.to_csv(f"lightwaves_uea_metrics_{scaling_method}_{dimension}.csv", mode='a',
                                            header=False,
                                            index=False)
                            progress_list.append(scaling_method + "_" + dimension + "_" + dataset + "_" + str(seed))
                            pickle.dump(progress_list, open("lightwaves_progress.pkl", 'wb'))
