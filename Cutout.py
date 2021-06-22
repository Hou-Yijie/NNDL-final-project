import pdb
import argparse
import csv
import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from torchvision import datasets, transforms


from utils import Cutout

from models import *
from utils import progress_bar

dataset_options = ['cifar10', 'cifar100']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', '-d', default='cifar10',
                    choices=dataset_options)
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate')
parser.add_argument('--seed', type=int, default=520,
                    help='random seed (default: 520)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')


parser.add_argument('--n_holes', type=int, default=1,
                    help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=16,
                    help='length of the holes')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
cudnn.benchmark = True  # Should make training should go faster for large models

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# Image Preprocessing
normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

train_transform = transforms.Compose([])
train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
train_transform.transforms.append(transforms.RandomHorizontalFlip())
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normalize)
#Cutout
train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))

test_transform = transforms.Compose([transforms.ToTensor(), normalize])

if args.dataset == 'cifar10':
    num_classes = 10
    train_dataset = datasets.CIFAR10(root='data/', train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(root='data/', train=False, transform=test_transform, download=True)

elif args.dataset == 'cifar100':
    num_classes = 100
    train_dataset = datasets.CIFAR100(root='data/', train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR100(root='data/', train=False, transform=test_transform, download=True)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                           shuffle=True, pin_memory=True, num_workers=4)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                          shuffle=False, pin_memory=True, num_workers=4)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Model
print('==> Building model..')
cnn = ResNet18()
cnn = cnn.to(device)


criterion = nn.CrossEntropyLoss().to(device)
cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.lr,
                                momentum=0.9, nesterov=True, weight_decay=args.decay)

logname = 'result/cutout' + '.csv'
if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'test acc'])


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

def test(loader):
    cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            pred = cnn(images)
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    cnn.train()
    return val_acc

if __name__=='__main__':
    for epoch in range(args.epochs):
        loss_avg = 0.
        correct = 0.
        total = 0.

        adjust_learning_rate(cnn_optimizer, epoch)

        progress_bar = tqdm(train_loader)
        for i, (images, labels) in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch))

            images = images.to(device)
            labels = labels.to(device)

            cnn.zero_grad()
            pred = cnn(images)

            loss = criterion(pred, labels)
            loss.backward()
            cnn_optimizer.step()

            loss_avg += loss.item()

            # Calculate running average of accuracy
            pred = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            correct += (pred == labels.data).sum().item()
            accuracy = correct / total

            progress_bar.set_postfix(
                loss='%.3f' % (loss_avg/ (i + 1)),
                acc='%.3f' % accuracy)

        test_acc = test(test_loader)
        tqdm.write('test_acc: %.3f' % (test_acc))


        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, loss_avg /(i+1), test_acc])

        print('Saving..')
        state = {
            'net': cnn.state_dict(),
            'acc': test_acc,
            'epoch': epoch,
        }
        torch.save(state, 'checkpoint/cutout' + '.pth.tar')