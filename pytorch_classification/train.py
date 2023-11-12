import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from dataset import get_train_loader_cifar100, get_val_loader_cifar100
from utils import get_network, WarmUpLR, most_recent_folder, \
            most_recent_weights, last_epoch, best_acc_weights

import pyzjr as pz
from pyzjr.dlearn import GPU_INFO

def train_one_epoch(trainingloader,epoch):
    time = pz.Timer()
    net.train()
    for batch_index, (images, labels) in enumerate(trainingloader):

        if args.Cuda:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(trainingloader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.batch_size + len(images),
            total_samples=len(trainingloader.dataset)
        ), end='\r', flush=True)

        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    time.stop()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, time.total()))

@torch.no_grad()
def eval_training(testloader,epoch=0, tb=True):

    time = pz.Timer()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in testloader:

        if args.Cuda:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    time.stop()
    if args.Cuda:
        GPU_INFO(
            headColor="red",
            gpuColor="blue"
        )
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(testloader.dataset),
        correct.float() / len(testloader.dataset),
        time.total()
    ), end='\r', flush=True)
    print()

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(test_loader.dataset), epoch)

    return correct.float() / len(test_loader.dataset)


if __name__ == '__main__':

    class parser_args():
        def __init__(self):
            self.net = "vgg16"
            self.Cuda = True
            self.EPOCHS = 100
            self.batch_size = 4
            self.warm = 1
            self.CHECKPOINT_PATH = 'checkpoint'
            self.resume = False
            self.lr = 0.01
            self.LOG_DIR = "logs"
            self.SAVE_EPOCH = 10
            self.MILESTONES = [60, 120, 160]
            self.DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
            self.TIME_NOW = datetime.now().strftime(self.DATE_FORMAT)
            self.CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
            self.CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

        def _help(self):
            stc = {
                "log_dir": "存放训练模型.pth的路径",
                "Cuda": "是否使用Cuda,如果没有GPU,可以使用CUP,i.e: Cuda=False",
                "EPOCHS": "训练的轮次,这里默认就跑100轮",
                "batch_size": "批量大小,一般为1,2,4",
                "warm": "控制学习率的'热身'或'预热'过程"
            }
            return stc
    args = parser_args()
    net = get_network(args)

    #data preprocessing:
    training_loader = get_train_loader_cifar100(
        args.CIFAR100_TRAIN_MEAN,
        args.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=True
    )

    test_loader = get_val_loader_cifar100(
        args.CIFAR100_TRAIN_MEAN,
        args.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=True
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(args.CHECKPOINT_PATH, args.net), fmt=args.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(args.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(args.CHECKPOINT_PATH, args.net, args.TIME_NOW)

    if not os.path.exists(args.LOG_DIR):
        os.mkdir(args.LOG_DIR)
    writerlog_path = pz.logdir(dir=args.LOG_DIR, format=True, prefix=args.net)
    writer = SummaryWriter(writerlog_path)
    input_tensor = torch.Tensor(1, 3, 32, 32)
    if args.Cuda:
        input_tensor = input_tensor.cuda()
    writer.add_graph(net, input_tensor)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(args.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(args.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(args.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(args.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(args.CHECKPOINT_PATH, args.net, recent_folder))


    for epoch in range(1, args.EPOCHS + 1):
        train_one_epoch(training_loader,epoch)
        acc = eval_training(test_loader,epoch)

        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue


        #start to save best performance model after learning rate decay to 0.01
        if epoch > args.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % args.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

    writer.close()
