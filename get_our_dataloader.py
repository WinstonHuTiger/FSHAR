import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from torchvision import transforms
y_label = {
  'jm': 0, 'run': 1, 'sit': 2, 'squat': 3, 'walk': 4 , 'rip': 5, 'throw': 6, 'wip': 7
}
class src_data_set(Dataset):
    def __init__(self, src_path = r"D:\dev_x\DataProc\output\111_200x100_dataset_std_halfed\env1",
                 action_list =  ['rip', 'run', 'sit', 'squat', 'walk'],
                 transform = None):
        self.src_path = src_path
        self.person_data_num = 15 * 2
        self.image_path = []
        self.label = []
        self.transform_ = transforms.Compose([
            transforms.Normalize()
        ])
        # act_list = os.listdir(self.src_path)


        for act in action_list:
            act_folder = os.path.join(self.src_path, act)
            npy_file_list = sorted(os.listdir(act_folder))
            # we only select the last 16 people
            for npy_file in npy_file_list[4 * self.person_data_num:]:
                self.image_path.append(os.path.join(act_folder, npy_file))
                temp_npy = np.zeros(len(y_label))
                temp_npy[y_label[act]] = 1
                self.label.append(temp_npy)

        self.label = np.array(self.label)
        self.label = np.array(torch.max(torch.Tensor(self.label), dim = 1)[1])
        self.transform = transform

    def __getitem__(self, item):
        image = np.load(self.image_path[item])
        label = self.label[item]
        if self.transform is None:
            image = self.transform_(image)
        return image, label