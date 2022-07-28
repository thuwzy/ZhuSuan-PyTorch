import torch
import torch.nn as nn

import math, os, sys

sys.path.append("..")


from zhusuan.framework.bn import  BayesianNet
from examples.utils import load_mnist_realval, save_img

class NICE(BayesianNet):
    def __init__(self, num_coupling, in_out_dim, mid_dim, hidden):
        super(NICE, self).__init__()
        self.in_out_dim = in_out_dim
