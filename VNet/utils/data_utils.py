import os
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader


def read_data_paths():
    pass


class Flare21Dataset(Dataset):
    def __init__(self, images_path, masks_path, transformer):
        super(Flare21Dataset, self).__init__()
        assert len(images_path) == len(masks_path)
        self.images_path = images_path
        self.masks_path = masks_path
        self.transformer = transformer

    def __getitem__(self, item):
        img_array = np.load(self.images_path[item])
        msk_array = np.load(self.masks_path[item])

    def __len__(self):
        return len(self.images_path)


def load_data():
    pass
