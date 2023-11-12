import os
import matplotlib
import torch
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

from torch.utils.tensorboard import SummaryWriter


class SummaryLossHistory():
    def __init__(self, log_dir, model, input_shape, batch=2, cuda=True):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.losses = []
        self.val_loss = []
        self.batch = batch
        self.writer = SummaryWriter(self.log_dir)
        try:
            input_tensor = torch.randn(self.batch, 3, input_shape[0], input_shape[1])
            if cuda:
                input_tensor = input_tensor.cuda()
            self.writer.add_graph(model, input_tensor)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss):
        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txts"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txts"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')
            plt.plot(iters, savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--', linewidth=2,
                     label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

    def close(self):
        self.writer.close()

