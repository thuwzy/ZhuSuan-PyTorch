#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Logistic-normal topic models using Monte-Carlo EM
Dense implementation, O(n_docs*n_topics*n_vocab)
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os
import time
from six.moves import range, zip
from copy import copy
import numpy as np
from zhusuan.mcmc import SGHMC, SGLD
from zhusuan.framework import BayesianNet
import zhusuan as zs
# from zhusuan.evaluation import AIS

from examples.utils import load_uci_bow

class LNTM(BayesianNet):
    def __init__(self, n_chains, n_docs, n_topics, n_vocab, eta_mean, eta_logstd):
        super(LNTM, self).__init__()
        self.n_chains = n_chains
        self.n_docs = n_docs
        self.n_topics = n_topics
        self.n_vocab = n_vocab
        self.eta_mean = eta_mean
        self.eta_logstd = eta_logstd

    def forward(self, x):
        pass

if __name__ == "__main__":
    # Load nips dataset
    data_name = 'nips'
    data_path = "./data/" + data_name + '.pkl.gz'
    X, vocab = load_uci_bow(data_name, data_path)
    # vocab size: 12419, num of docs: 1500
    training_size = 1200
    X_train = X[:training_size, :]
    X_test = X[training_size:, :]

    # Define model training parameters
    batch_size = 100
    n_topics = 50
    n_vocab = X_train.shape[1]
    n_chains = 1

    # Padding
    rem = batch_size - X_train.shape[0] % batch_size
    if rem < batch_size:
        X_train = np.vstack((X_train, np.zeros((rem, n_vocab))))

    sampler = SGLD(0.2)

    iters = X_train.shape[0] // batch_size
    Eta = np.zeros((n_chains, X_train.shape[0], n_topics), dtype=np.float32)
    Eta_mean = np.zeros(n_topics, dtype=np.float32)
    Eta_logstd = np.zeros(n_topics, dtype=np.float32)
