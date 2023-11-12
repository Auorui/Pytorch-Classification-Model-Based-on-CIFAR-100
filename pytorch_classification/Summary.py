from torchsummary import summary
from utils.utils import get_network

class parse_args():
    def __init__(self):
        self.net = "vgg16"
        self.num_class = 4
        self.Cuda = True
        self.shape = [224, 224]

args = parse_args()
net = get_network(args)
input_size = (3, args.shape[0], args.shape[1])
summary(net, input_size=input_size)
