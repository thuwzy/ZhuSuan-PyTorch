from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
from scipy import stats
import numpy as np
from zhusuan import mcmc
from zhusuan.framework import BayesianNet



class TestNode:
    def __init__(self, x):
        self.tensor = x


class Test_Model(BayesianNet):
    def __init__(self, x):
        super().__init__()
        self.nodes['x'] = TestNode(x)

    def forward(self, observed):
        self.observe(observed)
        return self

    def _log_joint(self, use_cache=False):
        x = self.observed['x']
        # x.requires_grad = True  # !check
        lh_noise = torch.normal(mean=0., std=2., size=x.shape)
        res = 2 * torch.pow(x, 2) - torch.pow(x, 4) + lh_noise
        return res.sum()




def sample_error_with(sampler, n_chains=1, n_iters=80000, thinning=50, burinin=None, dtype=torch.float32,
                      sampler_type='hmc'):
    if burinin is None:
        burinin = n_iters * 2 // 3
    x = torch.zeros([n_chains], dtype=dtype, requires_grad=True)
    model = Test_Model(x)
    samples = []
    for t in range(n_iters):
        if sampler_type == 'sgld':
            resample = True if t == 0 else False
            x_sample = sampler.sample(model, {}, resample)['x'].detach().numpy()
        else:
            x_sample = sampler.sample(model, {}, {'x': x})[0]['x'].detach().numpy()
        if np.isnan(x_sample.sum()):
            raise ValueError("nan encountered")
        if t >= burinin and t % thinning == 0:
            samples.append(x_sample)
    samples = np.array(samples)
    samples = samples.reshape(-1)
    A = 3
    xs = np.linspace(-A, A, 1000)
    pdfs = np.exp(2 * (xs ** 2) - xs ** 4)
    pdfs = pdfs / pdfs.mean() / A / 2
    est_pdfs = stats.gaussian_kde(samples)(xs)
    return np.abs(est_pdfs - pdfs).mean()


if __name__ == '__main__':
    sampler = mcmc.HMC(step_size=0.01, n_leapfrogs=20)
    e = sample_error_with(sampler, n_chains=100, n_iters=8000)
    print(e)
    assert (e < 0.008)