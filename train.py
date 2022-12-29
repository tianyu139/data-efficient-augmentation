import torchvision
import torch
import torch.nn as nn
import time

from models.resnet_imagenet import ResnetWrapper


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def get_model(arch, CLASS_NUM, pretrained=False, feature_extract=True):
    if arch == 'resnet50':
         if pretrained:
             model = torchvision.models.resnet50(pretrained=pretrained)
             set_parameter_requires_grad(model, feature_extract)
             num_ftrs = model.fc.in_features
             model.fc = nn.Linear(num_ftrs, CLASS_NUM)
             model = torch.nn.DataParallel(ResnetWrapper(model))
         else:
             model = torch.nn.DataParallel(ResnetWrapper(torchvision.models.resnet50(num_classes=CLASS_NUM, pretrained=pretrained)))

    elif arch == 'resnet18':
        if pretrained:
            model = torchvision.models.resnet18(pretrained=pretrained)
            set_parameter_requires_grad(model, feature_extract)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, CLASS_NUM)
            model = torch.nn.DataParallel(ResnetWrapper(model))
        else:
            model = torch.nn.DataParallel(ResnetWrapper(torchvision.models.resnet18(num_classes=CLASS_NUM, pretrained=pretrained)))
    elif arch in resnet_model_names:
        model = torch.nn.DataParallel(resnet.__dict__[arch](CLASS_NUM))
    model.cuda()
    return model


def train(train_loader, model, criterion, optimizer, half=False, return_loss=False):
    """
        Run one train epoch
    """
    print(f"Training - {len(train_loader.dataset)} examples")
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, weight) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        weight = weight.cuda()
        target_var = target
        if half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        loss = (loss * weight).mean()  # (Note)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    if return_loss:
        return data_time.sum, batch_time.sum, losses.avg
    else:
        return data_time.sum, batch_time.sum


def validate(val_loader, model, criterion, weights_per_class=None, half=False):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target, weights_per_class=weights_per_class)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


    return top1.avg, losses.avg


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


def accuracy(output, target, topk=(1,), weights_per_class=None):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        if weights_per_class is None:
            correct_k = correct[:k].view(-1).float().sum(0)
        else:
            weights = weights_per_class[target.cpu()]
            correct_k = (correct[:k].view(-1).float().cpu() * weights).sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
