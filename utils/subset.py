import utils.craig as craig
import numpy as np
import torch
import torch.nn as nn
import time


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_coreset(gradient_est,
                labels,
                N,
                B,
                num_classes,
                normalize_weights=True,
                gamma_coreset=0,
                smtk=0,
                st_grd=0,
                equal_num=True,
                replace=False,
                optimizer="LazyGreedy"):
    '''
    Arguments:
        gradient_est: Gradient estimate
            numpy array - (N,p)
        labels: labels of corresponding grad ests
            numpy array - (N,)
        B: subset size to select
            int
        num_classes:
            int
        normalize_weights: Whether to normalize coreset weights based on N and B
            bool
        gamma_coreset:
            float
        smtk:
            bool
        st_grd:
            bool

    Returns
    (1) coreset indices (2) coreset weights (3) ordering time (4) similarity time
    '''
    try:
        subset, subset_weights, _, _, ordering_time, similarity_time, cluster = craig.get_orders_and_weights(
            B,
            gradient_est,
            'euclidean',
            smtk=smtk,
            no=0,
            y=labels,
            stoch_greedy=st_grd,
            equal_num=equal_num,
            gamma=gamma_coreset,
            num_classes=num_classes,
            replace=replace,
            optimizer=optimizer)
    except ValueError as e:
        print(e)
        print(f"WARNING: ValueError from coreset selection, choosing random subset for this epoch")
        subset, subset_weights = get_random_subset(B, N)
        ordering_time = 0
        similarity_time = 0

    if normalize_weights:
        subset_weights = subset_weights / np.sum(subset_weights) * len(subset_weights)

    if len(subset) != B:
        print(f"!!WARNING!! Selected subset of size {len(subset)} instead of {B}")
    print(f'FL time: {ordering_time:.3f}, Sim time: {similarity_time:.3f}')

    return subset, subset_weights, ordering_time, similarity_time, cluster


def get_random_subset(B, N):
    print(f'Selecting {B} element from the random subset of size: {N}')
    order = np.arange(0, N)
    np.random.shuffle(order)
    subset = order[:B]
    subset_weights = np.ones(len(subset))

    return subset, subset_weights


def get_maxloss_subset(B, train_indexed_loader, model):
        criterion = torch.nn.CrossEntropyLoss(reduction='none').cuda()
        _, _, labels, logits = subsetlib.grad_predictions(train_indexed_loader, model, NUM_CLASSES, use_linear=False, return_logits_and_tensors=True)
        losses = criterion(logits, labels.long()).cpu().numpy()

        subset = np.argsort(losses)[-B:]
        subset_weights = np.ones(len(subset))

        return subset, subset_weights


def grad_predictions(loader, model, num_classes, use_linear=False, half_prec=False, return_logits_and_tensors=False):
    """
    Get predictions
    """
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()
    device = next(model.parameters()).device

    N = len(loader.dataset)
    embed_dim = model.module.emb_dim()
    logits = torch.zeros(N, num_classes).to(device)
    preds = torch.zeros(N, num_classes).to(device)
    labels = torch.zeros(N, dtype=torch.int).to(device)
    if use_linear:
        lasts = torch.zeros(N, embed_dim).to(device)

    end = time.time()
    with torch.no_grad():
        for i, (image, target, idx) in enumerate(loader):
            input_var = image.to(device)
            target = target.to(device)

            if half_prec:
                input_var = input_var.half()

            if use_linear:
                out, emb = model(input_var, use_linear=True)
                preds[idx, :] = nn.Softmax(dim=1)(out)
                lasts[idx, :] = emb
                logits[idx, :] = out
            else:
                out = model(input_var)
                preds[idx, :] = nn.Softmax(dim=1)(out)
                logits[idx, :] = out
            labels[idx] = target.int()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    preds = preds.cpu()
    labels = labels.cpu()
    logits = logits.cpu()
    if use_linear:
        lasts = lasts.cpu()
        g0 = preds - torch.eye(num_classes)[labels.long()]
        g0_expand = torch.repeat_interleave(g0, embed_dim, dim=1)
        g1 = g0_expand * lasts.repeat(1, num_classes)
        gradient_est = g1
    else:
        gradient_est = preds - (torch.eye(num_classes))[labels.long()]

    if return_logits_and_tensors:
        return gradient_est, preds, labels, logits
    else:
        return gradient_est.data.numpy(), preds.data.numpy(), labels.data.numpy()


