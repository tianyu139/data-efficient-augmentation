import os
import sys
sys.path.insert(0, '../')
sys.path.insert(0, './')
import torch
import torchvision
import torchvision.transforms as transforms
import dataset.datasets as datasets
import numpy as np
from torch.utils.data import Subset, Dataset
from train import get_model, train, validate
from dataset.autoaugment import ImageNetPolicy
from dataset.caltech import Caltech256, TransformedDataset
import utils.subset as subsetlib
import argparse
import pandas as pd
from dataset.cutout import Cutout
from augmix.imagenet import aug as AugMix


SUBSET_ALGOS = ['coreset', 'random', 'maxloss']


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('output_dir', type=str,
                        help='directory to output csv results')
    parser.add_argument('--epochs', default=40, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='whether to use pretrained model')
    parser.add_argument('--cutout', dest='cutout', action='store_true',
                        help='whether to use cutout augmentation')
    parser.add_argument('--augmix', dest='augmix', action='store_true',
                        help='whether to use augmix augmentation')
    parser.add_argument('--noise_only', dest='noise_only', action='store_true',
                        help='whether to use noise augmentation')
    parser.add_argument('--dataset', dest='dataset', type=str, default="caltech256", help='dataset to use')
    parser.add_argument('--arch', dest='arch', type=str, default="resnet18", help='model architecture')
    parser.add_argument('--subset_algo', dest='subset_algo', type=str, default="coreset", help='Subset algorithm', choices=SUBSET_ALGOS)
    parser.add_argument('--subset_size', '-s', dest='subset_size', type=float, help='size of the subset', default=0.5)
    parser.add_argument('--subset_main_dataset', dest='subset_main_dataset', action='store_true',
                        help='whether to subset the main dataset')
    parser.add_argument('--enable_subset_augment', dest='enable_subset_augment', action='store_true',
                        help='whether to enable subset augmentation')
    parser.add_argument('--enable_coreset_augment_weights', dest='enable_coreset_augment_weights', action='store_true',
                        help='whether to enable weights for coresets when using subset augmentation')
    parser.add_argument('--override_subset_main_random', dest='override_subset_main_random', action='store_true',
                        help='whether to override main subset with random subset')
    parser.add_argument('--use_linear', dest='use_linear', action='store_true', help='Linear layer for coreset gradient approximation')
    parser.add_argument('--no_equal_num', dest='no_equal_num', action='store_true',
                        help='whether to use equal num from each class in coreset selection')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', '-m', type=float, metavar='M', default=0.9,
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-b', '--batch_size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('-R', '--R', dest='R', type=int, metavar='R',
                        help='interval to select subset', default=1)



    args = parser.parse_args()

    dataset = args.dataset
    R = args.R
    subset_size = args.subset_size
    subset_algo = args.subset_algo
    subset_main_dataset = args.subset_main_dataset
    enable_subset_augment = args.enable_subset_augment
    enable_coreset_augment_weights = args.enable_coreset_augment_weights
    use_linear = args.use_linear
    epochs = args.epochs
    lr = args.lr
    momentum = args.momentum
    wd = args.weight_decay
    bs = args.batch_size
    pretrained = args.pretrained
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"Arguments:  {args}")

    if epochs == 40:
        milestones = np.array([20, 30])
    elif epochs == 90:
        milestones = np.array([30, 60])
    elif epochs == 150:
        milestones = np.array([80, 120])
    elif epochs == 200:
        milestones = np.array([100, 150])
    elif epochs == 400:
        milestones = np.array([250, 350])
    elif epochs == 1 or epochs == 0:
        # For testing purposes only
        milestones = np.array([0])
    else:
        raise Exception("Epoch schedule unknown")


    if args.dataset == "caltech256":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        TRAIN_TRANSFORMS = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
        ])

        if args.cutout:
            TRAIN_TRANSFORMS_STRONG = transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            Cutout(1, 32)
            ])
        elif args.augmix:
            TRAIN_TRANSFORMS_STRONG_PREPROCESS = transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
            ])
            TRAIN_TRANSFORMS_STRONG = lambda x: AugMix(x, TRAIN_TRANSFORMS_STRONG_PREPROCESS)
        elif args.noise_only:
            TRAIN_TRANSFORMS = transforms.Compose([
                    transforms.Resize(size=256),
                    transforms.CenterCrop(size=224),
                    transforms.ToTensor(),
                ])

            TRAIN_TRANSFORMS_STRONG = transforms.Compose([
                    transforms.Resize(size=256),
                    transforms.CenterCrop(size=224),
                    transforms.ColorJitter(brightness=0.2),
                    transforms.GaussianBlur(5),
                    transforms.ToTensor(),
            ])

        else:
            TRAIN_TRANSFORMS_STRONG = transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            ImageNetPolicy(),
                            transforms.ToTensor(),
            ])


        VAL_TRANSFORMS = transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
            ])
        NUM_CLASSES = 257
        train_set, test_set, train_set_indexed, class_weights, train_labels = datasets.get_caltech_datasets(root="./data", return_extra_train=True)

    elif args.dataset == "tinyimagenet":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        TRAIN_TRANSFORMS = transforms.Compose([
                transforms.RandomCrop(56),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
        ])

        TRAIN_TRANSFORMS_STRONG = transforms.Compose([
                transforms.RandomCrop(56),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(),
                transforms.ToTensor(),
                normalize
        ])
        VAL_TRANSFORMS = transforms.Compose([
                transforms.CenterCrop(size=56),
                transforms.ToTensor(),
                normalize
            ])
        NUM_CLASSES = 200
        train_set, test_set, train_set_indexed, class_weights, train_labels = datasets.get_tinyimagenet_datasets()

    elif args.dataset == "imagenet":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        TRAIN_TRANSFORMS = transforms.Compose([
                transforms.Resize(size=256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
        ])

        TRAIN_TRANSFORMS_STRONG = transforms.Compose([
                transforms.Resize(size=256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(),
                transforms.ToTensor(),
                normalize
        ])
        VAL_TRANSFORMS = transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                normalize
            ])
        NUM_CLASSES = 1000
        train_set, test_set, train_set_indexed, class_weights, train_labels = datasets.get_imagenet_datasets()


    model = get_model(args.arch, NUM_CLASSES, pretrained=pretrained, feature_extract=False)

    train_set = TransformedDataset(train_set, transform_default=TRAIN_TRANSFORMS, transform_strong=TRAIN_TRANSFORMS_STRONG)
    test_set = TransformedDataset(test_set, transform_default=VAL_TRANSFORMS, return_weight=False)
    train_set_indexed = TransformedDataset(train_set_indexed, transform_default=VAL_TRANSFORMS, return_weight=False, return_index=True)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=bs, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=bs, shuffle=False, num_workers=8)
    train_indexed_loader = torch.utils.data.DataLoader(
        train_set_indexed, batch_size=bs, shuffle=True, num_workers=8)

    print(f"Params: {sum(param.numel() for param in model.parameters())}    "
    f"Trainable params: {sum(param.numel() for param in model.parameters() if param.requires_grad)}")

    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay = wd)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, last_epoch=-1, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss(reduction='none').cuda()
    val_criterion = torch.nn.CrossEntropyLoss().cuda()

    N = len(train_set)
    B = int(N * subset_size)
    best_prec = 0
    best_epoch = 0
    prec = 0
    for epoch in range(epochs):
        with torch.no_grad():
            if subset_size < 1 and epoch % R == 0:
                if subset_algo == 'coreset':
                    gradient_est, preds, labels = subsetlib.grad_predictions(train_indexed_loader, model, NUM_CLASSES, use_linear=use_linear)
                    subset, subset_weights, _, _, _ = subsetlib.get_coreset(
                        gradient_est,
                        labels,
                        N=len(train_indexed_loader),
                        B=B,
                        num_classes=NUM_CLASSES,
                        normalize_weights=True,
                        gamma_coreset=0,
                        smtk=0,
                        st_grd=0,
                        equal_num=not args.no_equal_num,
                        replace=False,
                        optimizer="LazyGreedy")

                elif subset_algo == 'random':
                    subset, subset_weights = subsetlib.get_random_subset(B, N)
                elif subset_algo == 'maxloss':
                    subset, subset_weights = subsetlib.get_maxloss_subset(B, train_indexed_loader, model)

                if subset_main_dataset:
                    if args.override_subset_main_random and subset_algo != 'random':
                        print("Overriding main subset with random subset")
                        subset_rand, subset_weights_rand = subsetlib.get_random_subset(B, N)
                        train_loader.dataset.set_subset(subset_rand, subset_weights_rand)
                    else:
                        train_loader.dataset.set_subset(subset, subset_weights)
                if enable_subset_augment:
                    if subset_algo == 'coreset' and enable_coreset_augment_weights:
                        train_loader.dataset.set_augment_subset(subset, augment_subset_weights=subset_weights)
                    else:
                        train_loader.dataset.set_augment_subset(subset, augment_subset_weights=None)
            elif subset_size >= 1:
                if enable_subset_augment:
                    print("Augmenting full dataset...")
                    train_loader.dataset.set_augment_subset(np.arange(N), augment_subset_weights=None)
                else:
                    print("Full dataset + no augments")


        print(f"Epoch: {epoch}  Subset: {B} ", end='')
        train(train_loader, model, criterion, optimizer)
        lr_scheduler.step()
        with torch.no_grad():
            prec, _ = validate(val_loader, model, val_criterion, weights_per_class=class_weights)
            if prec > best_prec:
                best_prec = prec
                best_epoch = epoch
            print(f"Prec: {prec:.3f}  Best prec: {best_prec:.3f} @ Epoch {best_epoch}")



    result_dict = vars(args)
    result_dict['last_prec'] = prec
    result_dict['best_prec'] = best_prec
    result_dict['best_epoch'] = best_epoch

    df = pd.DataFrame([result_dict])
    counter = 0
    filename = os.path.join(output_dir, "result{}.csv")
    while os.path.isfile(filename.format(counter)):
        counter += 1
    filename = filename.format(counter)
    df.to_csv(filename, index=False, header=True)


if __name__ == '__main__':
    main()
