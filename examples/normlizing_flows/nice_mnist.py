import torch
import torch.nn as nn

import math, os, sys

sys.path.append("..")
import numpy as np
from zhusuan.framework.bn import BayesianNet
from zhusuan.distributions import Logistic, FlowDistribution
from zhusuan.invertible import get_coupling_mask, AdditiveCoupling, Scaling, RevSequential
from examples.utils import fetch_dataloaders, save_img, check_dir


class NICE(BayesianNet):
    def __init__(self, num_coupling, in_out_dim, mid_dim, hidden):
        super(NICE, self).__init__()
        self.in_out_dim = in_out_dim
        masks = get_coupling_mask(in_out_dim, 1, num_coupling)
        flow_layers = []
        for i in range(num_coupling):
            flow_layers.append(AdditiveCoupling(
                in_out_dim=in_out_dim,
                mid_dim=mid_dim,
                hidden=hidden,
                mask=masks[i]
            ))
        flow_layers.append(Scaling(in_out_dim))
        self.flow = RevSequential(flow_layers)
        dis = Logistic(loc=[0.], scale=[1.])
        self.sn(dis, name="prior_x")
        flow_dis = FlowDistribution(latents=self.nodes["prior_x"].dist, transformation=self.flow)
        self.sn(flow_dis, name="x", n_samples=-1)

    def sample(self, size):
        return self.nodes["x"].dist.sample(shape=[size, self.in_out_dim])

    def forward(self, x):
        z, log_det_J = self.flow.forward(x, reverse=False)
        log_ll = torch.sum(self.nodes["prior_x"].dist.log_prob(z[0]) + log_det_J, dim=1)
        return log_ll


def main():
    batch_size = 100
    epoch_size = 10
    sample_size = 64
    coupling = 10
    mask_config = 1.

    # Optim Parameters
    lr = 1e-3

    full_dim = 1 * 28 * 28
    mid_dim = 1000
    hidden = 10

    model = NICE(num_coupling=coupling,
                 in_out_dim=full_dim,
                 mid_dim=mid_dim,
                 hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-4)
    train_dataloader, test_dataloader = fetch_dataloaders('MNIST', batch_size, logit_transform=False, dequantify=True)

    for epoch in range(epoch_size):
        stats = []
        for _, data in enumerate(train_dataloader):
            model.train()
            optimizer.zero_grad()
            inputs = data[0]
            # loss = -model.nodes['x'].log_prob(inputs).mean()
            loss = -model(inputs).mean()
            loss.backward()
            optimizer.step()
            stats.append(loss.detach().numpy())
        print("Epoch:[{}/{}], Log Likelihood: {:.4f}".format(
            epoch + 1, epoch_size, np.mean(np.array(stats))
        ))

    model.eval()
    with torch.no_grad():
        samples = model.nodes['x'].dist.sample(shape=[sample_size, full_dim])
        samples = torch.reshape(samples, shape=[-1, 28 * 28])
        print(samples.shape)
        path = os.path.join(os.getcwd(), 'results', 'NICE')
        check_dir(path)
        save_img(samples.detach().cpu().numpy(), os.path.join(path, 'sample-NICE.png'))

if __name__ == '__main__':
    main()
