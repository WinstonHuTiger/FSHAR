import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from torchvision import transforms

y_label = set(['jm', 'run', 'sit', 'squat', 'walk', 'rip', 'throw', 'wip'])

envs = {
    "env1": (3.6637e-19, 0.99999),
    "env2": (-4.070817756958907e-19, 0.9999),
    "env3": (-3.0901207518733525e-18, 0.9999),
    "env4": (-5.273559366969494e-18, 1)

}
SEED = 2021


class data_set(Dataset):
    # For fine-tuning there are 3 or 12 samples used for each class
    def __init__(self, src_path=r"/data/sdb/qingqiao/stft_data/111_200x100_dataset_std_halfed",
                 env="env1", mode="train",
                 action_list=['rip', 'run', 'sit', 'squat', 'walk'],
                 transform=None, shots=3):
        self.person_data_num = 15 * 2
        self.image_path = []
        self.label = []

        assert (env in list(envs.keys()))
        assert set(action_list).issubset(y_label)
        self.mean_ = envs[env][0]
        self.std_ = envs[env][1]
        # act_list = os.listdir(self.src_path)
        self.src_path = os.path.join(src_path, env)
        self.fine_tuning_train_num = shots
        np.random.seed(SEED)

        for act in action_list:
            act_folder = os.path.join(self.src_path, act)

            npy_file_list = sorted(os.listdir(act_folder))
            # print(npy_file_list)
            if env != "env1":
                if mode == "train":
                    index = np.random.permutation(len(npy_file_list)).tolist()[:self.fine_tuning_train_num]
                else:
                    index = np.random.permutation(len(npy_file_list)).tolist()[self.fine_tuning_train_num:]
                npy_file_list = [npy_file_list[idx] for idx in index]
            else:
                # we only select the last 16 people
                npy_file_list = npy_file_list[4 * self.person_data_num:]
            for npy_file in npy_file_list:
                self.image_path.append(os.path.join(act_folder, npy_file))
                temp_npy = np.zeros(len(action_list))
                temp_npy[action_list.index(act)] = 1
                self.label.append(temp_npy)

        self.label = np.array(self.label)
        self.label = np.array(torch.max(torch.Tensor(self.label), dim=1)[1])
        # print(self.label)

        self.transform = transform

    def __getitem__(self, item):
        image = torch.FloatTensor(np.load(self.image_path[item]))
        label = self.label[item]
        # print(label)
        if self.transform is None:
            image = (image - self.mean_) / self.std_
        else:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_path)


def get_our_datasets(target_env="env2", shots=3,
                    #  test_action_list=['jm', 'run', 'sit', 'squat', 'walk', 'rip', 'throw', 'wip']
                     test_action_list=[ 'run', 'sit', 'squat', 'walk', 'rip']
                     ):
    source_dataset = data_set(env="env1")
    target_train_dataset = data_set(env=target_env, mode="train",
                                    action_list=test_action_list, shots=shots)
    target_test_dataset = data_set(env=target_env, mode="test",
                                   action_list=test_action_list, shots=shots)
    source_loader = DataLoader(source_dataset, batch_size=len(source_dataset), shuffle=True)
    target_train_loader = DataLoader(target_train_dataset, batch_size=len(target_train_dataset),
                                     shuffle=True)
    target_test_loader = DataLoader(target_test_dataset, shuffle=False,
                                    batch_size=len(target_test_dataset))

    return source_loader, target_train_loader, target_test_loader


if __name__ == "__main__":
    batch_size = 32
    source_loader, target_train_loader, target_test_loader = get_our_datasets(target_env="env3", shots=5)
    for x, y in target_train_loader:
        print(x.shape)
        print(y.shape)
    for x, y in target_test_loader:
        print(x.shape)
        print(y.shape)
        # print(y)
