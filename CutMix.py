# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py

import argparse
import os
import shutil
import time
import csv

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from models import *
import utils
import numpy as np

import warnings

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='Cutmix PyTorch CIFAR-10, CIFAR-100 and ImageNet-1k Training')

parser.add_argument('--dataset', dest='dataset', default='cifar10', type=str,
                    help='dataset (options: cifar10, cifar100, and imagenet)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--seed', default=520, type=int, help='random seed')

parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='to use basicblock for CIFAR datasets (default: bottleneck)')
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help='to print the status at every iteration')
parser.add_argument('--beta', default=1, type=float,
                    help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=1, type=float,
                    help='cutmix probability')

parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    global args
    args = parser.parse_args()

    if not os.path.isdir('result'):
        os.mkdir('result')
    logname = ('result/cutmix' + '.csv')

    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['epoch', 'train loss', 'test acc'])

    if args.seed != 0:
        torch.manual_seed(args.seed)

    #dataset
    if args.dataset.startswith('cifar'):
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        if args.dataset == 'cifar100':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('data/', train=True, download=True, transform=transform_train),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('data/', train=False, transform=transform_test),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            numberofclass = 100
        elif args.dataset == 'cifar10':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('data/', train=True, download=True, transform=transform_train),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('data/', train=False, download=True, transform=transform_test),
                batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
            numberofclass = 10
        else:
            raise Exception('unknown dataset: {}'.format(args.dataset))

    else:
        raise Exception('unknown dataset: {}'.format(args.dataset))

    #model
    # Model
    print('==> Building model..')
    model = ResNet18()
    model = model.to(device)
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)
    cudnn.benchmark = True

    for epoch in range(0, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        test_acc = validate(val_loader, model, criterion, epoch)

        with open(logname, 'a') as logfile:
           logwriter = csv.writer(logfile, delimiter=',')
           logwriter.writerow([epoch, train_loss, test_acc])

        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': test_acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/cutmix.pth.tar')



def train(train_loader, model, criterion, optimizer, epoch):
    print('\nEpoch: %d' % epoch)
    # switch to train mode
    model.train()

    train_loss = 0
    correct = 0
    total = 0

    for i, (input, target) in enumerate(train_loader):
        input = input.to(device)
        target = target.to(device)

        r = np.random.rand(1)
        if args.beta > 0 and r < args.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(input.size()[0]).to(device)
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            # compute output
            output = model(input)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
        else:
            # compute output
            output = model(input)
            loss = criterion(output, target)

        train_loss += loss.item()
        _, pred= torch.max(output.data, 1)
        total += target.size(0)
        correct += (lam * pred.eq(target_a.data).cpu().sum().float()
                    + (1 - lam) * pred.eq(target_b.data).cpu().sum().float())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return train_loss/(i+1)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def validate(val_loader, model, criterion, epoch):
    # switch to evaluate mode
    model.eval()

    test_loss = 0
    correct = 0
    total = 0

    for i, (input, target) in enumerate(val_loader):
        target = target.to(device)

        output = model(input)
        loss = criterion(output, target)

        test_loss += loss.item()
        _, pred = torch.max(output.data, 1)
        total += target.size(0)
        correct += pred.eq(target.data).cpu().sum()

    acc = 100. * correct / total
    return acc


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= 50:
        lr /= 10
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
