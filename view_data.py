import h5py
import os

root_path = "."
results_path = ["results_Cos", "results_SR"]
envs = ["env2", "env3", "env4"]

for p in results_path:
    print(" ===== {} ===== ".format(p))
    for k in [1, 3, 5]:
        for env in envs:
            for ac in [0, 1, 2]:
                print(" ===== shot_{} {} action_list{} ===== ".format( k, env, ac))
                data_path = os.path.join(root_path, p, env, "stftnum_samp_{}".format(k), "action_list_{}".format(ac), "accuracy_stat.h5")
                data = h5py.File(data_path, "r")
                acc = list(data["avg"])
                std = list(data["std"])
                print("{} acc: {}; std: {}".format(env, acc, std))


