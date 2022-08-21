import torch
import torch.nn as nn

import math, os, sys

sys.path.append("..")
import numpy as np
from zhusuan.framework.bn import BayesianNet
from zhusuan.distributions import Logistic, FlowDistribution
from zhusuan.invertible import get_coupling_mask, AdditiveCoupling, Scaling, RevSequential, Coupling
from examples.utils import fetch_dataloaders, save_img, check_dir
from nice_copy import prepare_data, StandardLogistic


class NICE(BayesianNet):
    def __init__(self, prior, num_coupling, in_out_dim, mid_dim, hidden):
        super(NICE, self).__init__()
        self.in_out_dim = in_out_dim
        masks = get_coupling_mask(in_out_dim, 1, num_coupling)
        flow_layers = []
        for i in range(num_coupling):
            flow_layers.append(Coupling(
                in_out_dim=in_out_dim,
                mid_dim=mid_dim,
                hidden=hidden,
                mask_config=(1.+i) % 2
            ))

        self.coupling = nn.ModuleList(flow_layers)
        self.scaling = Scaling(in_out_dim)
        # self.flow = RevSequential(flow_layers)
        # dis = Logistic(loc=[0.], scale=[1.])
        # self.sn(dis, name="prior_x")
        # flow_dis = FlowDistribution(latents=self.nodes["prior_x"].dist, transformation=self.flow)
        # self.sn(flow_dis, name="x", n_samples=-1)
        self.prior = prior


    def g(self, z):
        x, _ = self.scaling(z, reverse=True)
        for i in reversed(range(len(self.coupling))):
            x = self.coupling[i](x, reverse=True)[0]
        return x


    def f(self, x):
        for i in range(len(self.coupling)):
            x = self.coupling[i](x)[0]
        return self.scaling(x)

    def sample(self, size):
        z = self.prior.sample((size, self.in_out_dim))
        return self.g(z)
        # return self.nodes["x"].dist.sample(shape=[size, self.in_out_dim])

    def forward(self, x):
        z, log_det_J = self.f(x)
        # log_ll = torch.sum(self.nodes["prior_x"].dist.log_prob(z[0]), dim=1)
        log_ll = torch.sum(self.prior.log_prob(z), dim=1)
        return log_ll + log_det_J


def main():
    batch_size = 200
    epoch_size = 14
    sample_size = 64
    coupling = 4
    mask_config = 1.

    # Optim Parameters
    lr = 1e-3

    full_dim = 1 * 28 * 28
    mid_dim = 1000
    hidden = 5
    prior = StandardLogistic()
    model = NICE(prior=prior,
                 num_coupling=coupling,
                 in_out_dim=full_dim,
                 mid_dim=mid_dim,
                 hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-4)

    train_dataloader, test_dataloader = fetch_dataloaders('MNIST', batch_size, logit_transform=False, dequantify=True)
    num_iter = -1
    mean = torch.load('./mnist_mean.pt')
    # for i in model.parameters():
    #     print(i.shape)
    for epoch in range(epoch_size):
        stats = []
        for _, data in enumerate(train_dataloader):
            model.train()
            optimizer.zero_grad()
            inputs = data[0]
            inputs = inputs.reshape([batch_size, 1, 28, 28])
            inputs = prepare_data(
                inputs, 'mnist', zca=None, mean=mean)
            loss = -model(inputs).mean()
            loss.backward()
            optimizer.step()
            num_iter += 1
            stats.append(loss.detach().numpy())
            if num_iter % 1000 == 0:
                model.eval()
                with torch.no_grad():
                    # import IPython
                    # IPython.embed()
                    samples = model.sample(sample_size)
                    samples = prepare_data(
                        samples, 'mnist', zca=None, mean=mean, reverse=True)
                    samples = torch.reshape(samples, shape=[-1, 28 * 28])
                    path = os.path.join(os.getcwd(), 'results', 'NICE')
                    check_dir(path)
                    save_img(samples.detach().cpu().numpy(), os.path.join(path, 'sample-NICE2{}.png'.format(num_iter)))

        print("Epoch:[{}/{}], Log Likelihood: {:.4f} iter{}".format(
            epoch + 1, epoch_size, np.mean(np.array(stats)), num_iter
        ))

    model.eval()
    with torch.no_grad():
        samples = model.sample(sample_size)
        samples = prepare_data(
            samples, 'mnist', zca=None, mean=mean, reverse=True)
        samples = torch.reshape(samples, shape=[-1, 28 * 28])
        path = os.path.join(os.getcwd(), 'results', 'NICE')
        check_dir(path)
        save_img(samples.detach().cpu().numpy(), os.path.join(path, 'sample-NICE2.png'))

if __name__ == '__main__':
    main()
