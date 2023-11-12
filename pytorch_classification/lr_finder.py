import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
from utils import get_network
from dataset import get_train_loader
from pyzjr.dlearn.learnrate import lr_finder

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

if __name__ == '__main__':
    class parser_args():
        def __init__(self):
            self.net = "vgg16"
            self.batch_size = 64
            self.base_lr = 1e-7
            self.max_lr = 10
            self.num_iter = 100
            self.Cuda = True

    args = parser_args()

    mean = CIFAR100_TRAIN_MEAN
    std = CIFAR100_TRAIN_STD
    train_loader = get_train_loader(mean, std, batch_size=4)

    net = get_network(args)

    loss_function = nn.CrossEntropyLoss()
    lrfinder = lr_finder(net, train_loader, loss_function)
    lrfinder.update()
    lrfinder.plotshow()

    lrfinder.save(path="result.jpg")
