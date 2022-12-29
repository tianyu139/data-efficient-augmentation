import torch
import torchvision
import numpy as np
from torch.utils.data import Subset
from dataset.caltech import Caltech256
from dataset.tinyimagenet import TinyImageNet



def get_caltech_datasets(root="./data/", return_extra_train=True):
    NUM_TRAINING_SAMPLES_PER_CLASS = 60
    ds = Caltech256(root)

    class_start_idx = [0]+ [i for i in np.arange(1, len(ds)) if ds.y[i]==ds.y[i-1]+1]
    train_indices = sum([np.arange(start_idx,start_idx + NUM_TRAINING_SAMPLES_PER_CLASS).tolist() for start_idx in class_start_idx],[])
    test_indices = list((set(np.arange(1, len(ds))) - set(train_indices) ))
    train_set = Subset(ds, train_indices)
    test_set = Subset(ds, test_indices)
    train_set_2 = Subset(ds, train_indices)

    class_targets = np.array([ds.y[idx] for idx in test_indices])
    counts = np.unique(class_targets, return_counts=True)[1]
    class_weights = counts.sum()/(counts*len(counts))
    class_weights = torch.Tensor(class_weights)

    train_labels = np.array([ds.y[idx] for idx in train_indices])

    if return_extra_train:
        return train_set, test_set, train_set_2, class_weights, train_labels
    else:
        return train_set, test_set, class_weights, train_labels


def get_tinyimagenet_datasets(root="./data/tiny-imagenet-200", return_extra_train=True):
    train_set = TinyImageNet(root, split='train')
    test_set = TinyImageNet(root, split='val')
    train_set_2 = TinyImageNet(root, split='train')
    train_labels = train_set.y

    if return_extra_train:
        return train_set, test_set, train_set_2, None, train_labels
    else:
        return train_set, test_set, None, train_labels


def get_imagenet_datasets(root="./data/imagenet", return_extra_train=True):
    train_set = torchvision.datasets.ImageNet(root, split='train')
    test_set = torchvision.datasets.ImageNet(root, split='val')
    train_set_2 = torchvision.datasets.ImageNet(root, split='train')
    train_labels = train_set.targets

    if return_extra_train:
        return train_set, test_set, train_set_2, None, train_labels
    else:
        return train_set, test_set, None, train_labels
