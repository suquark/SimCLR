import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from data_aug.transforms import SimCLRTransformForRL

np.random.seed(0)


class RLTrajectoryDataSet(Dataset):
    def __init__(self, transform):
        self.transform = transform

    def __len__(self):
        # we only take the last 200,000 examples
        return (600_000 // 3) // 3

    def __getitem__(self, idx):
        imgs = []
        for i in range(3):
            im = Image.open(f"/home/ubuntu/efs/rad/saved_images/{400_000 + idx*3+i}.jpg")
            width, height = im.size   # Get dimensions
            new_width, new_height = 84, 84
            left = (width - new_width)/2
            top = (height - new_height)/2
            right = (width + new_width)/2
            bottom = (height + new_height)/2

            # Crop the center of the image
            im = im.crop((left, top, right, bottom))
            imgs.append(im)
        (x1, x1_params), (x2, x2_params) = self.transform(imgs)
        return (torch.cat(x1, dim=-3), torch.cat(x2, dim=-3)), (x1_params, x2_params)


class DataSetWrapper(object):

    def __init__(self, batch_size, num_workers, valid_size, input_shape, s):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.s = s
        self.input_shape = eval(input_shape)

    def get_data_loaders(self):
        data_augment = self._get_simclr_pipeline_transform()

        train_dataset = RLTrajectoryDataSet(transform=SimCLRDataTransform(data_augment))

        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def _get_simclr_pipeline_transform(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        data_transforms = SimCLRTransformForRL(self.input_shape[0], self.s)
        return data_transforms

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)
        return train_loader, valid_loader


class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj
