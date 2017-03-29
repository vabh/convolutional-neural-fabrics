from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.init as weight_init
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable

from utils import logger

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10', help='cifar10 | cifar100')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate, default=0.0002')
parser.add_argument('--cuda'  , action='store_false', help='enables cuda')
parser.add_argument('--save', help='folder to store log files, model checkpoints')

opt = parser.parse_args()
print(opt)

#logger
try:
    os.makedirs(opt.save)
    print('Logging at: ' + opt.save)
except OSError:
    pass
log = logger.Logger(opt.save+'/train.log', ['loss', 'train error', 'test error'])

# set random seed
opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if torch.cuda.is_available() and opt.cuda:
    torch.cuda.manual_seed(opt.manualSeed)

# set cudnn
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# get data loaders
# default set to cifar10
train_dataset = dset.CIFAR10(root=opt.dataroot, download=True, train=True,
                       transform=transforms.Compose([
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomCrop(32, padding=4),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
                       ]))
test_dataset = dset.CIFAR10(root=opt.dataroot, download=True, train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
                       ]))
assert train_dataset
assert test_dataset

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batchSize,
                                            shuffle=True, num_workers=int(opt.workers))

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=opt.batchSize,
                                            shuffle=False, num_workers=int(opt.workers))

# count number of incorrect classifications
def compute_score(output, target):
    pred = output.max(1)[1]
    incorrect = pred.ne(target).cpu().sum()
    batch_size = output.size(0)
    return incorrect


#define model

class UpSample(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(UpSample, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(outChannels)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.batch_norm(x)
        x = nn.ReLU(True)(x)
        return x

class DownSample(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=2, padding=1)
        self.batch_norm = nn.BatchNorm2d(outChannels)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = nn.ReLU(True)(x)
        return x

class SameRes(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(SameRes, self).__init__()
        self.conv = nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(outChannels)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = nn.ReLU(True)(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.channels = 128
        self.kernel_size = 3

        self.layers = 8
        self.scales = 5

        self.node_ops = nn.ModuleList()

        self.start_node = SameRes(3, self.channels)

        self.fc = nn.Linear(self.channels,10)

        for layer in range(self.layers):
            self.node_ops.append(nn.ModuleList()) # add list for each layer
            self.node_ops[layer] = nn.ModuleList() # list for each scale

            if layer == 0:
                for i in range(self.scales):
                    self.node_ops[layer][i] = nn.ModuleList()

                    node = DownSample(self.channels,self.channels)
                    self.node_ops[layer][i].append(node)
            else:
                for i in range(self.scales):
                    self.node_ops[layer][i] = nn.ModuleList()

                    node = SameRes(self.channels,self.channels)
                    self.node_ops[layer][i].append(node)
                    if i == 0:
                        self.node_ops[layer][i].append(
                            UpSample(self.channels,self.channels))
                    elif i == self.scales -1:
                        self.node_ops[layer][i].append(
                            DownSample(self.channels,self.channels))
                        if layer == self.layers-1:
                            self.node_ops[layer][i].append(
                                DownSample(self.channels,self.channels))
                    else:
                        self.node_ops[layer][i].append(
                            DownSample(self.channels,self.channels))
                        self.node_ops[layer][i].append(
                            UpSample(self.channels,self.channels))
                        if layer == self.layers-1:
                            self.node_ops[layer][i].append(
                                DownSample(self.channels,self.channels))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weight_init.kaiming_normal(m.weight)
                weight_init.constant(m.bias, 0.1)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0,0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        node_activ = [[[] for i in range(self.scales)] for j in range(self.layers)]
        out = self.start_node(x)
        for layer in range(self.layers):
            if layer == 0:
                for i in range(self.scales):
                    if i == 0:
                        node_activ[layer][i] = self.node_ops[layer][i][0](out)
                    else:
                        node_activ[layer][i] = self.node_ops[layer][i][0](node_activ[layer][i-1])
            else:
                for i in range(self.scales):
                    if i == 0:
                        t1 = (node_activ[layer-1][i])
                        t2 = self.node_ops[layer][i][1](node_activ[layer-1][i+1])
                        t = self.node_ops[layer][i][0](t1 + t2)
                        node_activ[layer][i] = t
                    elif i == self.scales-1:
                        t1 = (node_activ[layer-1][i])
                        t2 = self.node_ops[layer][i][1](node_activ[layer-1][i-1])

                        if layer == self.layers-1:
                            t3 = self.node_ops[layer][i][2](node_activ[layer][i-1])
                            t = self.node_ops[layer][i][0](t1 + t2  + t3)
                        else:
                            t = self.node_ops[layer][i][0](t1 + t2)
                        node_activ[layer][i] = t
                    else:
                        t1 = (node_activ[layer-1][i])
                        t2 = self.node_ops[layer][i][2](node_activ[layer-1][i+1])
                        t3 = self.node_ops[layer][i][1](node_activ[layer-1][i-1])
                        if layer == self.layers-1:
                            t4 = self.node_ops[layer][i][3](node_activ[layer][i-1])
                            t = self.node_ops[layer][i][0](t1 + t2 + t3 + t4)
                        else:
                            t = self.node_ops[layer][i][0](t1 + t2 + t3)
                        node_activ[layer][i] = t

        out = node_activ[-1][-1]
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out

net = Net()
# net.apply(weights_init)
print(net)

# criterion
criterion = nn.CrossEntropyLoss()

if opt.cuda:
    net.cuda()
    criterion.cuda()

# setup optimizer

#train
def train(epoch):
    net.train()
    score_epoch = 0
    loss_epoch = 0
    print('Epoch: ' + str(epoch))
    if epoch > 120:
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    elif epoch > 80:
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    else:
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)

    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images).cuda()
        labels = Variable(labels, requires_grad=False).cuda()

        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        loss_epoch = loss_epoch + loss.data[0]
        score_epoch = score_epoch + compute_score(output.data, labels.data)

    loss_epoch = loss_epoch / len(train_loader)
    print('[%d/%d][%d] train_loss: %.4f err: %d'
         % (epoch, opt.niter, len(train_loader), loss_epoch, score_epoch))
    return loss_epoch, score_epoch


#test network
def test():
    net.eval()
    score_epoch = 0
    loss_epoch = 0
    for i, (images, labels) in enumerate(test_loader):
        images = Variable(images).cuda()
        labels = Variable(labels, requires_grad=False).cuda()

        output = net(images)
        loss = criterion(output, labels)

        loss_epoch = loss_epoch + loss.data[0]
        score_epoch = score_epoch + compute_score(output.data, labels.data)

    loss_epoch = loss_epoch / len(test_loader)
    print('Test error: %d' % (score_epoch))
    return loss_epoch, score_epoch


#train for opt.niter epochs
start_error = test()
for epoch in range(1,opt.niter+1):
    train_loss, train_error = train(epoch)
    test_loss, test_error = test()
    log.add([train_loss, train_error/50000.0, test_error/10000.0])
    log.plot()
    if epoch % 10 == 0:
        # do checkpointing
        torch.save(net.state_dict(), '%s/net_epoch_%d.pth' % (opt.save, epoch))
