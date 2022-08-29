import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from zhusuan.framework.bn import BayesianNet
from zhusuan.distributions import Normal, Bernoulli
from zhusuan.variational.elbo import ELBO
from zhusuan.invertible import MaskCoupling, get_coupling_mask, Scaling, RevSequential
from examples.utils import load_mnist_realval, save_img

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Generator(BayesianNet):
    def __init__(self, x_dim, z_dim, batch_size):
        super(Generator, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.batch_size = batch_size

        self.gen = nn.Sequential(
            nn.Linear(z_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, x_dim)
        )

    def forward(self, observed):
        self.observe(observed)

        try:
            batch_len = self.observed['z'].shape[0]
        except:
            batch_len = 100

        prior = Normal(mean=torch.zeros([batch_len, self.z_dim]),
                       std=torch.ones([batch_len, self.z_dim]), dtype=torch.float32)
        z = self.sn(prior, "z")
        # print(z.shape)

        x_probs = self.gen(z)
        self.cache['x_mean'] = F.sigmoid(x_probs)

        dis = Bernoulli(probs=x_probs)
        sample_x = self.sn(dis, "x",
                           reduce_mean_dims=[0],
                           reduce_sum_dims=[1])
        assert (sample_x.shape[0] == batch_len)

        return self


class Variational(BayesianNet):
    def __init__(self, x_dim, z_dim, batch_size):
        super(Variational, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.batch_size = batch_size

        self.common_layer = nn.Sequential(
            nn.Linear(x_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU()
        )

        self.output_mean = nn.Linear(500, z_dim)
        self.output_sd = nn.Linear(500, z_dim)

    def forward(self, observed):
        self.observe(observed)
        x = self.observed['x']

        z_logits = self.common_layer(x)
        self.cache['z_logits'] = z_logits

        z_mean = self.output_mean(z_logits)
        z_sd = torch.exp(self.output_sd(z_logits))

        normal = Normal(mean=z_mean, std=z_sd, is_reparameterized=True)
        z = self.sn(normal, "z",
                    reduce_mean_dims=[0],
                    reduce_sum_dims=[1])
        return self


class NICEFlow(nn.Module):
    def __init__(self, z_dim, mid_dim, num_coupling, num_hidden):
        super(NICEFlow, self).__init__()
        masks = get_coupling_mask(z_dim, 1, num_coupling)
        flows = [MaskCoupling(
            in_out_dim=z_dim,
            mid_dim=mid_dim,
            hidden=num_hidden,
            mask=masks[i]
        ) for i in range(num_coupling)]
        flows.append(Scaling(z_dim))
        self.flow = RevSequential(flows)

    def forward(self,z, **kwargs):
        out, log_det_J = self.flow.forward(z, **kwargs)
        res = {"z": out}
        return res, log_det_J



class HF(nn.Module):
    def __init__(self, z_dim, is_first=False, v_dim=None):
        super(HF, self).__init__()
        if is_first:
            self.v_layer = nn.Linear(v_dim, z_dim)
        else:
            self.v_layer = nn.Linear(z_dim, z_dim)
    def forward(self, z, v, **kwargs):
        v_new = self.v_layer(v)
        vvT = torch.bmm(v_new.unsqueeze(2), v_new.unsqueeze(1))
        vvTz = torch.bmm(vvT, z.unsqueeze(2)).squeeze(2)
        norm_sq = torch.sum(v_new * v_new, dim=1, keepdim=True)
        #TODO

class HouseHolderFlow(nn.Module):
    def __init__(self, z_dim, v_dim, n_flows):
        super(HouseHolderFlow, self).__init__()


