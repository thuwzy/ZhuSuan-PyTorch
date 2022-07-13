import torch
import torch.nn as nn
import math, os, sys

sys.path.append("..")
from zhusuan.framework.bn import BayesianNet
from zhusuan.variational.importance_weighted_objective import ImportanceWeightedObjective
from examples.utils import load_mnist_realval, save_img

hidden_dim = 500


class Generator(BayesianNet):
    def __init__(self, x_dim, z_dim, n_samples):
        super(Generator, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.n_samples = n_samples
        self.gen_sq = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, x_dim),
            nn.Sigmoid()
        )

    def forward(self, observed: dict):
        """
        this step will add the observed (name:tensor) pair to the
        self._obseved dict
        """
        self.observe(observed)
        raise NotImplementedError()


class Variational(BayesianNet):
    def __init__(self, x_dim, z_dim, n_samples):
        super(Variational, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.n_samples = n_samples
        # define common forward layer
        self.output_logits = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # each of them output the mean and logstd of posterior of q(z|x)
        self.output_mean = nn.Linear(hidden_dim, z_dim)
        self.output_logstd = nn.Linear(hidden_dim, z_dim)

    def forward(self, observed):
        self.observe(observed)
        x = self.observed['x']
        z_logits = self.output_logits(x)
        z_mean = self.output_mean(z_logits)
        z_logstd = self.output_logstd(z_logits)
        z_std = torch.exp(z_logstd)
        # get samples from StochasticTensor, z is not observed, so self.sn return new samples
        # z.shape [n_samples, batch_shape, z_dim]
        z = self.sn("Normal",
                    name="z",
                    mean=z_mean,
                    std=z_std,
                    reparameterize=True,
                    n_samples=self.n_samples,
                    reduce_mean_dims=None,
                    reduce_sum_dims=[2]
                    )
        return self
