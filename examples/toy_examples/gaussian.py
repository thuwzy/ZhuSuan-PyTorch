#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import torch
from scipy import stats
import matplotlib.pyplot as plt
import zhusuan as zs
from zhusuan.framework import BayesianNet
from zhusuan.distributions import  Normal
def gaussian(n_x, stdev, n_partivles):
    bn = BayesianNet()
    dist = Normal(torch.zeros([n_x]), std=stdev)
    bn.sn(dist, "x", n_samp)