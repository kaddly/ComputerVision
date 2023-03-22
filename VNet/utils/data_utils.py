import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.transforms import get_transform


def read_data_paths(data_dir='Images'):
    train_path, val_path = [], []
    for i in os.listdir('../data/' + data_dir):
        if i.split('_')[0] == 'testing':
            val_path.append(os.path.join('../data', data_dir, i))
        else:
            train_path.append(os.path.join('../data', data_dir, i))
    return train_path, val_path


class Flare21Dataset(Dataset):
    def __init__(self, images_path, masks_path, transforms):
        super(Flare21Dataset, self).__init__()
        assert len(images_path) == len(masks_path)
        self.images_path = images_path
        self.masks_path = masks_path
        self.transforms = transforms

    def __getitem__(self, item):
        img_array = np.load(self.images_path[item]).astype(np.float32)
        msk_array = np.load(self.masks_path[item]).astype(np.int8)
        img = torch.FloatTensor(img_array).permute(2, 1, 0).unsqueeze(0)
        msk = torch.tensor(msk_array).permute(2, 1, 0)/4

        img, msk = self.transforms(img, msk)

        return img, (msk*4).long()

    def __len__(self):
        return len(self.images_path)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=0)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


def load_data(batch_size):
    img_train_path, img_val_path = read_data_paths('Images')
    msk_train_path, msk_val_path = read_data_paths('Masks')

    train_transforms = get_transform(train=True)
    train_dataset = Flare21Dataset(img_train_path, msk_train_path, train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_train_path = get_transform(train=False)
    val_dataset = Flare21Dataset(img_val_path, msk_val_path, val_train_path)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader
