import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import h5py
from main_ours import main_env2env_fs

"""ours"""

tgt_num_samp_per_class_list = [3]
K = 4
num_exp = 3
tgtsbj_list = [3, 5, 12]
src_type_list = ["stft"]
sim_type_list = ["SR", "Cos"]
target_action_lists  = [['jm', 'run', 'sit', 'squat', 'walk', 'rip', 'throw', 'wip'],
                        ['jm', 'throw', 'wip']]
target_envs = ["env2", "env3", "env4"]

# TGT_NUM_CLASSES = 8
#
# accuracy_list_general = np.zeros((num_exp, K))
# accuracy_list_general_avg = np.zeros((num_exp))
# accuracy_list_general_std = np.zeros((num_exp))
#
# confusemat_list_general = np.zeros((TGT_NUM_CLASSES, TGT_NUM_CLASSES, num_exp, K))
# confusemat_list_general_avg = np.zeros((TGT_NUM_CLASSES, TGT_NUM_CLASSES, num_exp))

count = -1
for env in target_envs:
    print("===== {} =====".format(env))
    for ii, action_list in enumerate(target_action_lists):
        print("====== action_list_{} ======".format(ii))
        for sim_type in sim_type_list:
            for src_type in src_type_list:
                for tgt_num_samp_per_class in tgt_num_samp_per_class_list:
                    TGT_NUM_CLASSES = len(action_list)
                    accuracy_list_general = np.zeros((num_exp, K))
                    accuracy_list_general_avg = np.zeros((num_exp))
                    accuracy_list_general_std = np.zeros((num_exp))

                    confusemat_list_general = np.zeros((TGT_NUM_CLASSES, TGT_NUM_CLASSES, num_exp, K))
                    confusemat_list_general_avg = np.zeros((TGT_NUM_CLASSES, TGT_NUM_CLASSES, num_exp))
                    rslt_path = os.path.join("results_" + sim_type,
                                            env,
                                              src_type + "num_samp_" + str(tgt_num_samp_per_class),
                                              "action_list_" + str(ii))
                    if not os.path.exists(rslt_path):
                        os.makedirs(rslt_path)
                    if not os.path.exists( os.path.join(rslt_path,"accuracy_stat.h5")):
                        for k in range(K):
                            k = int(k)
                            print('k = ' + str(k))
                            accuracy_list, confusemat_list = \
                                main_env2env_fs(tgt_num_samp_per_class, src_type, sim_type, k, env, action_list)
                            accuracy_list_general[:, k] = accuracy_list.tolist()
                            confusemat_list_general[:, :, :, k] = confusemat_list
                            print(accuracy_list_general)
                        for i in range(num_exp):
                            accuracy_list_general_avg[i] = \
                                np.mean(np.squeeze(accuracy_list_general[i, :])).tolist()
                            accuracy_list_general_std[i] = \
                                np.std(np.squeeze(accuracy_list_general[i, :])).tolist()
                            confusemat_list_general_avg[:, :, i] = \
                                np.mean(confusemat_list_general[:, :, i, :], axis=2)
                            print("The result for the " + str(i) + "th accuracy is: " + \
                                  str(int(round(accuracy_list_general_avg[i]))) + "+\-" + \
                                  str(int(round(accuracy_list_general_std[i]))))
                        # Save statistics as plot
                        count = count + 1
                        plt.figure(count)
                        plt.errorbar(list(range(0, num_exp)), accuracy_list_general_avg,
                                     accuracy_list_general_std, marker='s', mfc=None,
                                     mec='blue', ms=5, mew=2)
                        plt.xlim((-0.1, num_exp + 0.1))
                        plt.savefig(os.path.join(rslt_path, "accuracy_stat.png"))
                        plt.clf()
                        plt.close()
                        # Save statistics as file
                        f = h5py.File(os.path.join(rslt_path, "accuracy_stat.h5"), "w")
                        f.create_dataset("avg", data=accuracy_list_general_avg)
                        f.create_dataset("std", data=accuracy_list_general_std)
                        f.close()
                        print('Done.')
                        # Save all results as file
                        f = h5py.File(os.path.join(rslt_path, "accuracy.h5"), "w")
                        f.create_dataset("list", data=accuracy_list_general)
                        f.close()
                        print('Done.')
                        # Save statistics as file
                        f = h5py.File(os.path.join(rslt_path, "confusemat_stat.h5"), "w")
                        f.create_dataset("avg", data=confusemat_list_general_avg)
                        f.close()
                        print('Done.')
